import torch
import torch.nn as nn

## Defining layers
linear = nn.Linear(in_features=4096, out_features=10)
conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=1)  # Channels are modified, but the size of the image is the same.
relu = nn.ReLU(False)

# Test of linear
print("===================")
x = torch.randn(4096)
y = linear(x)
print(f'size of y after linear: {y.size()}')

# Test of conv and relu
print("===================")
x = torch.randn(1, 3, 7, 7)
print(f'size of x (initial): {x.size()}')
y = conv(x)
print(f'size of y after conv: {y.size()}')
z = relu(x)
print(f'size of z after relu: {z.size()}')

# List of tuples
for name, p in linear.named_parameters():
    print(name,'. Its size:',p.size(), end='\n')
