import torch
import os
from spikingjelly.datasets import dvs128_gesture
from spikingjelly.datasets import n_mnist
import spikingjelly.datasets as sd

def load_dvs128_gesture(distributed=False, T=10, data_dir = 'dvs128_gesture'):
    train_set = dvs128_gesture.DVS128Gesture(data_dir, train=True, data_type='frame', frames_number=T,
                                                 split_by='number')
    test_set = dvs128_gesture.DVS128Gesture(data_dir, train=False, data_type='frame', frames_number=T,
                                                 split_by='number')
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.SequentialSampler(test_set)
    return train_set, test_set, train_sampler, test_sampler

def load_n_mnist(distributed=False, T=10, data_dir = 'n_mnist'):
    train_set = n_mnist.NMNIST(data_dir, train=True, data_type='frame', split_by='time', frames_number=T)
    test_set = n_mnist.NMNIST(data_dir, train=False, data_type='frame', split_by='time', frames_number=T)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.SequentialSampler(test_set)
    return train_set, test_set, train_sampler, test_sampler

if __name__ == '__main__':
    # os.makedirs('n_mnist', exist_ok=True)
    # os.makedirs('n_mnist/download', exist_ok=True)
    train_set, test_set, _, _ = load_n_mnist(distributed=False, T=10, data_dir = '/data1/dsk/dataset/n_mnist')
    print(train_set[0][0].shape)
