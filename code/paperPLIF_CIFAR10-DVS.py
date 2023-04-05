import sys
#sys.path.append('/content/drive/MyDrive/Colab Notebooks/ELAP/SNN/spikingjelly/')
import os
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.activation_based.neuron import BaseNode, LIFNode
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import math

class PLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan(), detach_reset=True):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def neuronal_charge(self, dv: torch.Tensor):
        if self.v_reset is None:
            # self.v += dv - self.v * self.w.sigmoid()
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            # self.v += dv - (self.v - self.v_reset) * self.w.sigmoid()
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        
    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'

class CIFAR10DVSNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()

        conv = []
        number_layer = 4
       

        for i in range(number_layer):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels
        
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(PLIFNode())
            conv.append(layer.MaxPool2d(2,2))
        
        h, w = 128, 128
        h = h >> number_layer
        w = w >> number_layer

        self.conv_fc = nn.Sequential(
                            *conv,
                            
                            layer.Flatten(),
                            layer.Dropout(0.5),
                            layer.Linear(channels*h*w, channels*h*w//4, bias=False),
                            PLIFNode(),

                            layer.Dropout(0.5),
                            layer.Linear(channels*h*w//4, 10*10), #1 num_classes = 10
                            PLIFNode(),

                            layer.VotingLayer(10)
                        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

def main():

    parser = argparse.ArgumentParser(description='Classify CIFAR 10 DVS')

    parser.add_argument('-init_tau', type=float, default=2.0)
    parser.add_argument('-T', default = 16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=4, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-j', default=8, type=int, metavar='N', help='number of data loading workers (default:8)')
    parser.add_argument('-data-dir', type=str, default='./datasets/CIFAR10DVS', help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd','adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-save-es', default=None, help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}')

    args = parser.parse_args()
    print(args)

    net = CIFAR10DVSNet(channels=args.channels)

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance = neuron.ParametricLIFNode)
    
    print(net)

    net.to(args.device)

    train_set = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')
    #print(train_set)
    test_set = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')


    train_data_loader = torch.utils.data.DataLoader(
                            dataset = train_set,
                            batch_size = args.b,
                            shuffle = True,
                            drop_last = True,
                            num_workers = args.j,
                            pin_memory = True
                        )

    test_data_loader = torch.utils.data.DataLoader(
                            dataset = test_set,
                            batch_size = args.b,
                            shuffle = True,
                            drop_last = False,
                            num_workers = args.j,
                            pin_memory = True
                        )
    
    scaler = None

    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer =None 
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum)
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'CIFAR10DVSNet_PLIF_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_b{args.b}_ep{args.epochs}')

    if args.amp:
        out_dir += '_amp'
    
    if args.cupy:
        out_dir += '_cupy'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    
    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()
            
            train_samples += label.numel()
            train_loss += loss.item()*label.numel()
            train_acc += (out_fr.argmax(1)==label).float().sum().item()

            functional.reset_net(net)
        
        train_time = time.time()
        train_speed = train_samples/(train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item()*label.numel()
                test_acc += (out_fr.argmax(1)==label).float().sum().item()
                functional.reset_net(net)
        
        test_time = time.time()
        test_speed = test_samples/(test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')




if __name__ == '__main__':
    main()