# RIAI 2021 Course Project



## Folder structure
In the directory `code` you can find 2 files. 
File `networks.py` contains encoding of fully connected neural network architectures as PyTorch classes.
The architectures extend `nn.Module` object and consist of standard PyTorch layers and a new SPU activation function. Please note that first layer of each network performs normalization of the input image.
File `verifier.py` contains a template of verifier. Loading of the stored networks and test cases is already implemented in `main` function. If you decide to modify `main` function, please ensure that parsing of the test cases works correctly. Your task is to modify `analyze` function by building upon DeepPoly convex relaxation. Note that provided verifier template is guaranteed to achieve **0** points (by always outputting `not verified`).

In folder `mnist_nets` you can find 10 neural networks (total of 5 architectures and 2 different models per architecture). These networks are loaded using PyTorch in `verifier.py`.
In folder `test_cases` you can find 10 subfolders. Each subfolder is associated with one of the networks, using the same name. In a subfolder corresponding to a network, you can find 2 test cases for this network. 
As explained in the lecture, these test cases **are not** part of the set of test cases which we will use for the final evaluation. 

## Setup instructions

We recommend you to install Python virtual environment to ensure dependencies are same as the ones we will use for evaluation.
To evaluate your solution, we are going to use Python 3.7.
You can create virtual environment and install the dependencies using the following commands:

```bash
$ virtualenv venv --python=python3.7
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the verifier

We will run your verifier from `code` directory using the command:

```bash
$ python verifier.py --net {net} --spec ../test_cases/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `net0_fc1, net1_fc1, net0_fc2, net1_fc2, net0_fc3, net1_fc3, net0_fc4, net1_fc4, net0_fc5, net1_fc5`.
`test_idx` is an integer representing index of the test case, while `eps` is perturbation that verifier should certify in this test case.

To test your verifier, you can run for example:

```bash
$ python verifier.py --net net0_fc1 --spec ../test_cases/net0_fc1/example_img0_0.01800.txt
```

To evaluate the verifier on all networks and sample test cases, we provide the evaluation script.
You can run this script using the following commands:

```bash
chmod +x evaluate
./evaluate
```
