python -u paperPLIF_CIFAR10-DVS.py -T 8 -device 'cuda:0' > ./logs/printlog_paperPLIF_CIFAR10DVS_1.txt 2>&1 &
python -u paperPLIF_CIFAR10-DVS.py -T 20 -device 'cuda:1' > ./logs/printlog_paperPLIF_CIFAR10DVS_2.txt 2>&1 &
python -u paperPLIF_CIFAR10-DVS.py -T 8 -channels 16 -device 'cuda:2' > ./logs/printlog_paperPLIF_CIFAR10DVS_3.txt 2>&1 &&
python -u paperPLIF_CIFAR10-DVS.py -T 20 -channels 16 -device 'cuda:0' > ./logs/printlog_resnet18_CIFAR10.txt 2>&1 &    