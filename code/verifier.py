import argparse
import torch
from networks import FullyConnected

from transformers import DeepPoly, DPLinear, DPSPU
import networks
import torch.nn as nn
import torch.optim as optim

DEVICE = 'cpu'
INPUT_SIZE = 28


def verifier_network(net, pixel_values):
    # build the verifier network
    layers = [module for module in net.modules() if type(module) not in [networks.FullyConnected, nn.Sequential]]
    verifier_layers = []

    for layer in layers:
        if type(layer) == networks.SPU:
            if len(verifier_layers) == 0:
                verifier_layers.append(DPSPU(len(pixel_values)))
            else:
                verifier_layers.append(DPSPU(verifier_layers[-1].out_features))
        if type(layer) == nn.Linear:
            verifier_layers.append(DPLinear(layer))

    return nn.Sequential(*verifier_layers)


def analyze(net, inputs, eps, true_label):
    pixel_values = inputs.view(-1)
    mean = 0.1307
    sigma = 0.3081
    '''transformer of the normalization layer'''
    lb = ((pixel_values - eps).clamp(0, 1) - mean) / sigma
    ub = ((pixel_values + eps).clamp(0, 1) - mean) / sigma
    verifier_net = verifier_network(net, pixel_values)
    opt = optim.Adam(verifier_net.parameters(), lr=1)
    num_iter = 1000
    for i in range(num_iter):
        opt.zero_grad()
        verifier_output = verifier_net(DeepPoly(lb.shape[0], lb, ub))
        verify_result = verifier_output.compute_verify_result(true_label)
        if (verify_result > 0).all():
            # print("Success after the ", i, "th iteration !")
            return True
        if i == num_iter - 1:
            return False
        loss = torch.log(-verify_result[verify_result < 0]).max()
        loss.backward()
        opt.step()


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
