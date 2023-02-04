import sys
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
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import math

class PLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=surrogate.ATan(), monitor_state=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            # self.v += dv - self.v * self.w.sigmoid()
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            # self.v += dv - (self.v - self.v_reset) * self.w.sigmoid()
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        return self.spiking()

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'

def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, use_plif, use_max_pool, detach_reset):
    # 首层是in_channels-out_channels
    # 剩余number_layer - 1层都是out_channels-out_channels
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)


def create_2fc(channels, h, w, dpp, class_num, init_tau, use_plif, detach_reset):
    return nn.Sequential(
        nn.Flatten(),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
        PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
    )


class NeuromorphicNet(nn.Module):
    def __init__(self, T, init_tau, use_plif, use_max_pool, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        out_spikes_counter = self.boost(self.fc(self.conv(x[0])).unsqueeze(1)).squeeze(1)
        for t in range(1, x.shape[0]):
            out_spikes_counter += self.boost(self.fc(self.conv(x[t])).unsqueeze(1)).squeeze(1)
        return out_spikes_counter

class DVS128GestureNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                        use_max_pool=use_max_pool, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=11,
                            init_tau=init_tau, use_plif=use_plif, detach_reset=detach_reset)


def main():
    
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')

    parser.add_argument('-init_tau', type=float, default=2.0)
    parser.add_argument('-T', default = 16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:1', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-j', default=8, type=int, metavar='N', help='number of data loading workers (default:8)')
    parser.add_argument('-data-dir', type=str, default='./datasets/DVSGesture/', help='root dir of MNIST dataset')
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

    net = DVS128GestureNet(T = args.T, init_tau=args.init_tau, use_plif=True, use_max_pool=True, detach_reset = True, channels=args.channels, number_layer=8)

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance = neuron.ParametricLIFNode)
    
    print(net)

    net.to(args.device)

    train_set = DVS128Gesture(root=args.data_dir, train = True, data_type='frame', frames_number=args.T, split_by='number')
    #print(train_set)
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')


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

    out_dir = os.path.join(args.out_dir, f'DVSNet_PLIF_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_b{args.b}_ep{args.epochs}')

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
            label_onehot = F.one_hot(label, 11).float()

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
                label_onehot = F.one_hot(label, 11).float()
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