import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepPoly:
    def __init__(self, size, lb, ub):
        """
        slb: symbolic lower bound
        sub: symbolic upper bound
        lb: concrete lower bound
        ub: concrete upper bound
        """
        self.slb = torch.cat([torch.diag(torch.ones(size)), torch.zeros(size).unsqueeze(1)], dim=1)
        self.sub = self.slb
        self.lb = lb
        self.ub = ub
        self.history = []
        self.layers = 0
        self.is_spu = False

    def save(self):
        """ Save all constraints for the back substitution """
        lb = torch.cat([self.lb, torch.ones(1)])
        ub = torch.cat([self.ub, torch.ones(1)])
        if self.is_spu:
            # spu layer
            slb = self.slb
            sub = self.sub
        else:
            # other layers
            keep_bias = torch.zeros(1, self.slb.shape[1])
            keep_bias[0, self.slb.shape[1] - 1] = 1
            slb = torch.cat([self.slb, keep_bias], dim=0)
            sub = torch.cat([self.sub, keep_bias], dim=0)
        # layer num
        self.layers += 1
        # record each layer
        self.history.append((slb, sub, lb, ub, self.is_spu))
        return self

    def resolve(self, constrains, layer, lower=True):
        """
        lower = True: return the lower bound
        lower = False: return the upper bound
        """
        # distinguish the sign of the coefficients of the constraints
        pos_coeff = F.relu(constrains)
        neg_coeff = F.relu(-constrains)
        layer_info = self.history[layer]
        is_spu = layer_info[-1]
        if layer == 0:
            # layer_info[2],layer_info[3]: concrete lower and upper bound
            lb, ub = layer_info[2], layer_info[3]
        else:
            # layer_info[0],layer_info[1]: symbolic lower and upper bound
            lb, ub = layer_info[0], layer_info[1]
        if not lower:
            lb, ub = ub, lb
        if is_spu:
            lb_diag, lb_bias = lb[0], lb[1]
            ub_diag, ub_bias = ub[0], ub[1]
            lb_bias = torch.cat([lb_bias, torch.ones(1)])
            ub_bias = torch.cat([ub_bias, torch.ones(1)])

            m1 = torch.cat([pos_coeff[:, :-1] * lb_diag, torch.matmul(pos_coeff, lb_bias).unsqueeze(1)], dim=1)
            m2 = torch.cat([neg_coeff[:, :-1] * ub_diag, torch.matmul(neg_coeff, ub_bias).unsqueeze(1)], dim=1)
            return m1 - m2
        else:
            return torch.matmul(pos_coeff, lb) - torch.matmul(neg_coeff, ub)

    def compute_verify_result(self, true_label):
        self.save()
        n = self.slb.shape[0] - 1
        unit = torch.diag(torch.ones(n))
        weights = torch.cat((-unit[:, :true_label], torch.ones(n, 1), -unit[:, true_label:], torch.zeros(n, 1)), dim=1)

        for i in range(self.layers, 0, -1):
            weights = self.resolve(weights, i - 1, lower=True)

        return weights


'''Transformer of affine layer'''


class DPLinear(nn.Module):
    def __init__(self, nested: nn.Linear):
        super().__init__()
        self.weight = nested.weight.detach()
        self.bias = nested.bias.detach()
        self.in_features = nested.in_features
        self.out_features = nested.out_features

    def forward(self, x):
        x.save()
        # append bias as last column
        init_slb = torch.cat([self.weight, self.bias.unsqueeze(1)], dim=1)
        x.lb = init_slb
        x.ub = init_slb
        x.slb = init_slb
        x.sub = init_slb
        for i in range(x.layers, 0, -1):
            x.lb = x.resolve(x.lb, i - 1, lower=True)
            x.ub = x.resolve(x.ub, i - 1, lower=False)
        x.is_spu = False
        return x


'''Transformer of SPU layer'''


class DPSPU(nn.Module):
    def __init__(self, in_features):
        super(DPSPU, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.lam = torch.nn.Parameter(torch.ones(in_features))
        # self.beta = torch.nn.Parameter(torch.ones(in_features))
        self.gamma = torch.nn.Parameter(torch.ones(in_features))
        self.theta = torch.nn.Parameter(torch.ones(in_features))

    @staticmethod
    def spu(x):
        return torch.where(x > 0, x ** 2 - 0.5, torch.sigmoid(-x) - 1)

    @staticmethod
    def derivative_sigmoid_part(x):
        # derivative of the sigmoid part of spu
        return -torch.sigmoid(x) * (1 - torch.sigmoid(x))

    def forward(self, x):
        x.save()
        lb, ub = x.lb, x.ub
        # different cases
        mask_1, mask_2 = lb.ge(0), ub.le(0)
        mask_3 = ~(mask_1 | mask_2)

        '''
        lb > 0
        '''
        # optimization 1:
        # The range of lower bound's slope: 2lb <= k1 <= 2ub
        slope_lower_1 = 2 * lb + 2 / (1 + torch.exp(self.lam)) * (ub - lb)
        bias_lower_1 = self.spu(slope_lower_1 / 2) - slope_lower_1 ** 2 / 2
        slope_upper_1 = ub + lb
        bias_upper_1 = self.spu(lb) - slope_upper_1 * lb

        '''
        ub < 0
        '''
        slope_lower_2 = (self.spu(lb) - self.spu(ub)) / (lb - ub)
        bias_lower_2 = self.spu(lb) - slope_lower_2 * lb
        # optimization 2:
        # The range of upper bound's slope: derivative(lb) <= k2 <= 0
        # slope_upper_2 = 1 / (1 + torch.exp(self.beta)) * derivative_sigmoid_part(lb)
        slope_upper_2 = self.derivative_sigmoid_part(lb)
        bias_upper_2 = self.spu(lb) - slope_upper_2 * lb

        '''
        lb < 0 < ub
        '''
        slope_upper_3 = (self.spu(ub) - self.spu(lb)) / (ub - lb)
        bias_upper_3 = self.spu(ub) - slope_upper_3 * ub
        derivative_lb = self.derivative_sigmoid_part(lb)
        bias_tangent = self.spu(lb) - derivative_lb * lb
        slope_upper_3_final = torch.where(derivative_lb < slope_upper_3, slope_upper_3, derivative_lb)
        bias_upper_3_final = torch.where(derivative_lb < slope_upper_3, bias_upper_3, bias_tangent)

        slope_negative = (self.spu(lb) - (-0.5)) / (lb - 0)
        slope_positive = 2 * ub
        slope_lower_3_final = slope_negative + 1 / (1 + torch.exp(self.gamma)) * (slope_positive - slope_negative)
        bias_lower_3_final = torch.where(slope_lower_3_final > 0, (-slope_lower_3_final ** 2 / 4 - 0.5),
                                         -0.5 * torch.ones(slope_lower_3_final.shape[0]))

        '''
        symbolic upper bound & lower bound
        '''
        curr_slb = slope_lower_1 * mask_1 + slope_lower_2 * mask_2 + slope_lower_3_final * mask_3
        curr_slb_bias = bias_lower_1 * mask_1 + bias_lower_2 * mask_2 + bias_lower_3_final * mask_3
        curr_sub = slope_upper_1 * mask_1 + slope_upper_2 * mask_2 + slope_upper_3_final * mask_3
        curr_sub_bias = bias_upper_1 * mask_1 + bias_upper_2 * mask_2 + bias_upper_3_final * mask_3

        x.lb = self.spu(lb) * mask_1 + self.spu(ub) * mask_2 - 0.5 * mask_3
        x.ub = self.spu(ub) * mask_1 + self.spu(lb) * mask_2 + (
            torch.where(self.spu(ub) > self.spu(lb), self.spu(ub), self.spu(lb))) * mask_3
        x.slb = torch.cat([curr_slb.unsqueeze(0), curr_slb_bias.unsqueeze(0)], dim=0)
        x.sub = torch.cat([curr_sub.unsqueeze(0), curr_sub_bias.unsqueeze(0)], dim=0)
        x.is_spu = True
        return x
