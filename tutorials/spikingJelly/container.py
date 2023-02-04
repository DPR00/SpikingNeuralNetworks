import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer

## Container

### The major containers in SpikingJelly are:
###  - multi_step_forward in functional style and MultiStepContainer in module style
###  - seq_to_ann_forward in functional style and SeqToANNContainer in module style
###  - StepModeContainer for wrapping a single-step module for single/multi-step propagation

### multi_step_forward can use a single-step module to implement multi-step propagation, and
### MultiStepContainer can wrap a single-step module to a multi-step module. For example:

net_s = neuron.IFNode(step_mode='s')
T, N, C, H, W = 4, 1, 3, 8, 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = functional.multi_step_forward(x_seq, net_s)
print(f'y_seq shape: {y_seq.shape}') # y_seq.shape = [T, N, C, H, W]

net_s.reset()
net_m = layer.MultiStepContainer(net_s) ## get WARNING
z_seq = net_m(x_seq)
print(f'z_seq shape: {z_seq.shape}') # z_seq.shape = [T, N, C, H, W]

### For a stateless ANN layer such as torch.nn.Conv2d, which requires input data with shape = [N, *],
### to be used in multi-step mode, we can wrap it by the multi-step containers:
print("----------------------------------")
with torch.no_grad():
    T, N, C, H, W = 4, 1, 3, 8, 8
    x_seq = torch.rand([T, N, C, H, W])

    conv = nn.Conv2d(C, 8, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(8)

    y_seq = functional.multi_step_forward(x_seq, (conv, bn))
    print(f'y_seq.shape={y_seq.shape}')

    net = layer.MultiStepContainer(conv, bn)
    z_seq = net(x_seq)
    print(f'z_seq.shape={z_seq.shape}')

### However, the ANN layers are stateless and Y[t] is only determined by X[t]. Hence, 
### it is not necessary to calculate step-bt-step. We can use seq_to_ann_forward or SeqToANNContainer
### to wrap, which will reshape the input with shape = [T, N, *] to shape = [TN, *],
### send data to ann layers, and reshape output to shape = [T, N, *]. The calculation in
### different time-steps are in parallelism and faster:
print("----------------------------------")
with torch.no_grad():
    T, N, C, H, W = 4, 1, 3, 8, 8
    x_seq = torch.rand([T, N, C, H, W])

    conv = nn.Conv2d(C, 8, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(8)

    y_seq = functional.multi_step_forward(x_seq, (conv, bn))
    print(f'y_seq.shape={y_seq.shape}')  # y_seq.shape = [T, N, C, H, W]

    net = layer.MultiStepContainer(conv, bn)
    z_seq = net(x_seq)
    print(f'z_seq.shape={z_seq.shape}')  # z_seq.shape = [T, N, C, H, W]

    p_seq = functional.seq_to_ann_forward(x_seq, (conv, bn))
    print(f'p_seq.shape={p_seq.shape}')  # p_seq.shape = [T, N, C, H, W]

    net = layer.SeqToANNContainer(conv, bn)
    q_seq = net(x_seq)
    print(f'q_seq.shape={q_seq.shape}')  # q_seq.shape = [T, N, C, H, W]

    # y_seq, z_seq, p_seq are q_seq are identical

### Most frequently-used ann modules have been defined in spikingjelly.activation_based.layer.
### It is recommended to use modules in spikingjelly.activation_based.layer, rather than using
### a container to wrap the ann layers manually. Althouth the modules in
### spikingjelly.activation_based.layer are implementd by using seq_to_ann_forward to wrap
### forward function, the advantages of modules in spikingjelly.activation_based.layer are:
###     - Both single-step and multi-step modes are supported. When using SeqToANNContainer
###       or MultiStepContainer to wrap modules, only the multi-step mode is supported.
###     - The wrapping of containers will add a prefix of keys() of state_dict, which brings
###       some troubles for loading weights.
### For example:
print("----------------------------------")
ann = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1, bias = False),
    nn.BatchNorm2d(8),
    nn.ReLU()
)

print(f'ann.state_dict.keys()={ann.state_dict().keys()}')

net_container= nn.Sequential(
    layer.SeqToANNContainer(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
    ),
    neuron.IFNode(step_mode='m')
)

net_origin = nn.Sequential(
    layer.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(8),
    neuron.IFNode(step_mode='m')
)

print(f'net_container.state_dict.keys()={net_container.state_dict().keys()}')

try:
    print('net_container is trying to load state dict from ann ...')
    net_container.load_state_dict(ann.state_dict())
    print('Load success!')
except BaseException as e:
    print('net_container can not load! The error message is\n', e)

try:
    print('net_origin is trying to load state dict from ann ...')
    net_origin.load_state_dict(ann.state_dict())
    print('Loadd success!')
except BaseException as e:
    print('net_origin can not load! The error message is', e)

### MultiStepContainer and SeqToANNContainer only support for
### multi-step mode and do not allow to switch to single-step mode.


### StepModeContainer works like the merged version of MultiStepContainer and SeqToANNContainer,
### which can be used to wrap stateless or stateful single-step modules.The user should specify
### whether the wrapped modules are stateless or stateful when using this container.
### This container also supports switching step modes.
### Here is an example of wrapping a stateless layer:
print("----------------------------------")
with torch.no_grad():
    T, N, C, H, W = 4, 1, 3, 8, 8
    x_seq = torch.rand([T, N, C, H, W])

    net = layer.StepModeContainer(
        False,
        nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(C),
    )
    net.step_mode = 'm'
    y_seq = net(x_seq)
    print(f'y_seq.shape={y_seq.shape}')  # y_seq.shape = [T, N, C, H, W]

    net.step_mode = 's'
    y = net(x_seq[0])
    print(f'y.shape={y.shape}')  # y_seq.shape = [N, C, H, W]


### Here is an example of wrapping a stateful layer:
print("----------------------------------")
with torch.no_grad():
    T, N, C, H, W = 4, 2, 4, 8, 8
    x_seq = torch.rand([T, N, C, H, W])

    net = layer.StepModeContainer(
        True,
        neuron.IFNode() # WARNING
    )

    net.step_mode = 'm'
    y_seq = net(x_seq)
    print(f'y_seq.shape={y_seq.shape}')  # y_seq.shape = [T, N, C, H, W]

    functional.reset_net(net)

    net.step_mode = 's'
    y = net(x_seq[0])
    print(f'y.shape={y.shape}')  # y_seq.shape = [T, N, C, H, W]

    functional.reset_net(net)

### It is safe to use set_step_mode to change the step mode of StepModeContainer.
### Only the step_mode of the container itself is changed, and the modules inside
### the container still use single-step: 
print("----------------------------------")
with torch.no_grad():
    net = layer.StepModeContainer(
        True,
        neuron.IFNode() #WARNING
    )
    print(net)
    functional.set_step_mode(net, 'm')
    print(f'net.step_mode={net.step_mode}')
    print(f'net[0].step_mode={net[0].step_mode}')


### In most cases, we use MultiStepContainer or StepModeContainer to wrap modules
### which do not define the multi-step forward, such as a network layer that exists
### in torch.nn but does not exist in spikingjelly.activation_based.layer.




