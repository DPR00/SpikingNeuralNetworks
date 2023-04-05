# import torch
# import torch.nn as nn
# from spikingjelly.activation_based import surrogate, neuron, functional
# from spikingjelly.activation_based.model import spiking_resnet

# s_resnet18 = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

# print(f's_resnet18={s_resnet18}')

# with torch.no_grad():
#     T = 4
#     N = 1
#     x_seq = torch.rand([T, N, 3, 224, 224])
#     functional.set_step_mode(s_resnet18, 'm')
#     y_seq = s_resnet18(x_seq)
#     print(f'y_seq.shape={y_seq.shape}')
#     functional.reset_net(s_resnet18)


from spikingjelly.datasets.es_imagenet import ESImageNet

root_dir = './datasets/ESImagenet'
train_set = ESImageNet(root_dir, train=True, data_type='event')
#test_set = ESImageNet(root_dir, train=False, data_type='frame')
