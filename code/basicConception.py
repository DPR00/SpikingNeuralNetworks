import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer

# Basic conception

## Activation-based Representation
print("---------------------------")
v = torch.rand([8])
v_th = 0.5 # Threshhold
### spikingjelly.activation_based uses tensors whose element
### is only 0 or 1 to represent spikes. For example:
spike = (v_th <= v).to(v) 
print('spike = ', spike)

## Data format

### There are two formats:
###  - Single time-step with shape = [N, *]. N: batch dimension. *: extra dim
###  - Many time-steps with shape = [T, N, *]. T: time-step dimension.

## Step Mode

### Single-step mode ('s'). Data's shape: [N, *]
### Multi-step mode ('m'). Data's shape: [T, N, *]
### Step mode can be set at "__init__" or change 'step_mode' after is built.
print("---------------------------")
net = neuron.IFNode(step_mode='m') # M-sm
net.step_mode = 's' # S-sm
print(net)

### If we want to input the sequence data with shape = [T, N, *]
### to a single-step module, we need to implement a for-loop
### in time-steps manually, which splits the sequence data into T data with
### shape = [N, *] and sends the data step-by-step.
print("---------------------------")
net_s = neuron.IFNode(step_mode='s')
T, N, C, H, W = 4, 1, 3, 8, 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = []

for t in range(T):
    x = x_seq[t] # x.shape = [N, C, H, W]
    y = net_s(x) # y.shape = [N, C, H,, W]
    y_seq.append(y.unsqueeze(0))

y_seq = torch.cat(y_seq)
print(y_seq.shape)

### Also, multi_step_forward wraps the for-loop in time-steps
### for single-step modules to handle sequence data with shape = [T, N, *],
### which is more convenient to use:
print("---------------------------")
net_s = neuron.IFNode(step_mode='s')
T, N, C, H, W = 4, 1, 3, 8, 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = functional.multi_step_forward(x_seq, net_s)
print(y_seq.shape)

### However, the best usage is to set the module as a multi-step module directly:
print("---------------------------")
net_m = neuron.IFNode(step_mode='m')
T, N, C, H, W = 4, 1, 3, 8, 8
x_seq = torch.rand([T, N, C, H, W])
y_seq = net_m(x_seq)
print(y_seq.shape)

### For compatibility with older versions,  the default step mode for all modules
### in SpikingJelly is single-step.

## Saving and Resetting of States

### Similar to RNN, Y[t] = f(X[t], H[t-1])
### In PyTorch, RNN outputs not only Y but also H. Different from PyTorch, the states
### are stored inside the module in spikingjelly.activation_based. For example, let us
### create a new layer of IF neurons, set them to single-step mode, and check
### the default voltage before and after giving inputs:
print("---------------------------")
net_s = neuron.IFNode(step_mode='s') # v = 0.0 by default
x = torch.rand([4])
print(net_s)
print(f'the initial v={net_s.v}')
y = net_s(x)
print(f'x={x}')
print(f'y={y}')
print(f'v={net_s.v}')

### If we give a new input sample, we should clear the previous states of the neurons
### and reset the neurons to the initialization states, which can be done by calling
### the module’s self.reset() function:
print("---------------------------")
net_s = neuron.IFNode(step_mode='s') # v = 0.0 by default
x = torch.rand([4])
print(f'check point 0: v={net_s.v}')
y = net_s(x)
print(f'check point 1: v={net_s.v}')
net_s.reset()
print(f'check point 2: v={net_s.v}')
x = torch.rand([8])
y = net_s(x)
print(f'check point 3: v={net_s.v}')

### For convenience, we can also call spikingjelly.activation_based.functional.reset_net
### to reset all modules in a network.

### If the network uses one or more stateful modules, it must be reset after processing
### one batch of data during training and inference:

# from spikingjelly.activation_based import functional
# # ...
# for x, label in tqdm(train_data_loader):
#     # ...
#     optimizer.zero_grad()
#     y = net(x)
#     loss = criterion(y, label)
#     loss.backward()
#     optimizer.step()

#     functional.reset_net(net)
#     # Never forget to reset the network!

### If we forget to reset, we may get a wrong output during inference or an error during training:

### RuntimeError: Trying to backward through the graph a second time
### (or directly access saved variables after they have already been freed).
### Saved intermediate values of the graph are freed when you call .backward() or autograd.grad().
### Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved variables after calling backward.


## Prapagation patterns

### If all modules in a network are single-step modules, the computation graph of the
### entire network is built step-by-step. Similar to single step mode

### If all modules in a network are multi-step modules, the computation graph of the
### entire network is built layer-by-layer. For example:
print("---------------------------")
T, N, C = 4, 2, 8
x_seq = torch.rand([T, N, C])*64

net = nn.Sequential(
    layer.Linear(C, 4),
    neuron.IFNode(),
    layer.Linear(4, 2),
    neuron.IFNode()
)

functional.set_step_mode(net, step_mode='m')

with torch.no_grad():
    y_seq_layer_by_layer = x_seq
    #print(net.__len__())
    for i in range(net.__len__()):
        #print(net[i])
        y_seq_layer_by_layer = net[i](y_seq_layer_by_layer) # iterate over each layer

### In most cases, we don’t need an explicit implementation of for i in range(net.__len__()),
### because torch.nn.Sequential has already done that for us. So, we write codes in the
### following simple style:

y_seq_layer_by_layer = net(x_seq)

### The only difference between step-by-step and layer-by-layer is the building order
### of the computation graph, and their outputs are identical:
print("---------------------------")
T, N, C, H, W = 4, 2, 3, 8, 8
x_seq = torch.rand([T, N, C, H, W])*64

net = nn.Sequential(
    layer.Conv2d(3, 8, kernel_size=3, padding=1, stride=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    neuron.IFNode(),
    layer.Flatten(start_dim=1),
    layer.Linear(8*H//2*W//2, 10),
    neuron.IFNode(),    
)

print(f'net={net}')

with torch.no_grad():
    y_seq_step_by_step = []
    for t in range(T):
        x = x_seq[t]
        y = net(x)
        y_seq_step_by_step.append(y.unsqueeze(0))
    
    y_seq_step_by_step = torch.cat(y_seq_step_by_step, 0)
    # we can also use `y_seq_step_by_step = functional.multi_step_forward(x_seq, net)`
    # to get the same results

    print(f'y_seq_step_by_step=\n{y_seq_step_by_step}')

    functional.reset_net(net)
    functional.set_step_mode(net, step_mode='m')
    y_seq_layer_by_layer = net(x_seq)

    max_error = (y_seq_layer_by_layer - y_seq_step_by_step).abs().max()
    print(f'max_error={max_error}') # 0.0. So, identical output


## Recommendations
 
### Although the difference is only in the building order of the computation graph
### (step-by-step an layer-by-layer), there are still some slight differences in
### computation speed and memory consumption of the two propagation patterns.

### - When using the surrogate gradient method to train SNN directly,
###    it is recommended to use the layer-by-layer propagation pattern.
###    When the network is built correctly, the layer-by-layer propagation
###    pattern has the advantage of parallelism and speed.
###
### - Using step-by-step propagation pattern when memory is limited.
###   For example, a large T is required in the ANN2SNN task. In the layer-by-layer
###   propagation pattern, the real batch size for stateless layers is TN rather than N
###   (refer to the next tutorial). when T is too large, the memory consumption may be too large.