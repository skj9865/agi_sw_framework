import os
import warnings
import random
import torchvision.datasets

import tonic
from tonic import DiskCachedDataset

import torch
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as datasets
from timm.data import ImageDataset, create_loader, Mixup, FastCollateMixup, AugMixDataset
from timm.data import create_transform, distributed_sampler
from timm.data.loader import PrefetchLoader
from tonic import DiskCachedDataset
from torchvision import transforms
import torchaudio
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from transform import Identity, Roll, Rotate, Scale, DropEventChunk, Jitter1D, OneHotLabels, cut_mix_augmentation

def get_shd_data(dir, batch_size, step, num_workers=16, **kwargs):
    """
    获取SHD数据
    https://ieeexplore.ieee.org/abstract/document/9311226
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    :format: (b,t,c,len) 不同于vision, audio中c为1, 并且没有h,w; 只有len=700
    """
    sensor_size = tonic.datasets.SHD.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])

    train_dataset = tonic.datasets.SHD(dir, transform=train_transform, train=True)

    test_dataset = tonic.datasets.SHD(dir, transform=test_transform, train=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader


def get_shd_data_aug1(dir, batch_size, step, num_workers=16, **kwargs):
    sensor_size = tonic.datasets.SHD.sensor_size
    portion = 0.15
    times=2
    train_transform = transforms.Compose([
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ] + [    
        torchaudio.transforms.FrequencyMasking(freq_mask_param=int(sensor_size[0]*portion), iid_masks=False),
        torchaudio.transforms.TimeMasking(time_mask_param=int(step*portion), iid_masks=False)
    ]*times
    )
    
    test_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ])

    train_dataset = tonic.datasets.SHD(dir, transform=train_transform, train=True)

    test_dataset = tonic.datasets.SHD(dir, transform=test_transform, train=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader


def get_shd_data_aug2(dir, batch_size, step, 
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        num_workers=16, **kwargs):
    sensor_size = tonic.datasets.SHD.sensor_size
    portion = 0.15
    times=2
    train_transform = transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ] + [
        
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ] + [    
        torchaudio.transforms.FrequencyMasking(freq_mask_param=int(sensor_size[0]*portion), iid_masks=False),
        torchaudio.transforms.TimeMasking(time_mask_param=int(step*portion), iid_masks=False)
    ]*times
    )
    
    test_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ])

    train_dataset = tonic.datasets.SHD(dir, transform=train_transform, train=True)

    test_dataset = tonic.datasets.SHD(dir, transform=test_transform, train=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader


def get_ssc_data(dir, batch_size, step, num_workers=16, **kwargs):
    sensor_size = tonic.datasets.SSC.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])

    train_dataset = tonic.datasets.SSC(dir, transform=train_transform, split='train')

    test_dataset = tonic.datasets.SSC(dir, transform=test_transform, split='test')


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader


def get_ssc_data_aug1(dir, batch_size, step, num_workers=16, **kwargs):
    sensor_size = tonic.datasets.SSC.sensor_size
    portion = 0.15
    times=2
    train_transform = transforms.Compose([
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ] + [    
        torchaudio.transforms.FrequencyMasking(freq_mask_param=int(sensor_size[0]*portion), iid_masks=False),
        torchaudio.transforms.TimeMasking(time_mask_param=int(step*portion), iid_masks=False)
    ]*times
    )
    
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ])

    train_dataset = tonic.datasets.SSC(dir, transform=train_transform, split='train')

    test_dataset = tonic.datasets.SSC(dir, transform=test_transform, split='test')


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader






def get_spiking_audio_datasets(dataset:str, dir:str, batch_size:int, n_time_bins:int, n_freq_bins:int,
        input_method: str = 'sum', #sum, resize
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        sa_portion: float = 0.15,
        sa_times: int = 2,
        fp16: bool = False,
        num_workers:int = 16, 
        pin_memory: bool = True,
        **kwargs):
    
    def nearest_multiple(x: int, target: int) -> int:
        assert (x > 0) and (target > 0) and (x < target)
        
        quotient = target // x
        lower_multiple = x * quotient
        upper_multiple = x * (quotient + 1)
        
        return lower_multiple if abs(target - lower_multiple) <= abs(target - upper_multiple) else upper_multiple


    if dataset == 'shd':
        sensor_size = tonic.datasets.SHD.sensor_size
    elif dataset == 'ssc':
        sensor_size = tonic.datasets.SSC.sensor_size
    else:
        raise
    
    
    #[Ch,F,T]
    print(f'dataset input_mothod="{input_method}"')
    if input_method == 'sum':
        print(f'nearest_multiple(n_freq_bins={n_freq_bins}, sensor_size[0]={sensor_size[0]})={nearest_multiple(n_freq_bins, sensor_size[0])}')
        input_method_transform = [
            transforms.Resize(size=[nearest_multiple(n_freq_bins, sensor_size[0]), n_time_bins]),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,F,T]->[Ch,T,F]
            transforms.Lambda(lambda x: x.reshape(1, n_time_bins, n_freq_bins, -1).sum(dim=-1, keepdim=False)),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,T,F]->[Ch,F,T]
        ]
    elif input_method == 'resize':
        input_method_transform = [
            transforms.Resize(size=[n_freq_bins, n_time_bins])
        ]
    else:
        raise
    
    train_transform = [
        # tonic transforms
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ] + [
        # tonic 2 torch
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ] + input_method_transform + [
        # torch transforms (specaug)
        torchaudio.transforms.FrequencyMasking(freq_mask_param=int(n_freq_bins*sa_portion), iid_masks=False),
        torchaudio.transforms.TimeMasking(time_mask_param=int(n_time_bins*sa_portion), iid_masks=False)
    ]*sa_times
    
    test_transform = [
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ] + input_method_transform
    
    
        
    if fp16:
        train_transform += [transforms.Lambda(lambda x: x.half())]
        test_transform += [transforms.Lambda(lambda x: x.half())]
        
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    

    
    if dataset == 'shd':
        train_dataset = tonic.datasets.SHD(dir, transform=train_transform, train=True)
        test_dataset = tonic.datasets.SHD(dir, transform=test_transform, train=False)
    elif dataset == 'ssc':
        train_dataset = tonic.datasets.SSC(dir, transform=train_transform, split='train')
        test_dataset = tonic.datasets.SSC(dir, transform=test_transform, split='test')
    else:
        raise


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=pin_memory, drop_last=True, num_workers=num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=pin_memory, drop_last=False, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader

def get_spiking_audio_datasets_v2(dataset:str, dir:str, batch_size:int, n_time_bins:int, n_freq_bins:int,
        input_method: str = 'sum_bilinear', #sum_bilinear, sum_bicubic, resize_bilinear, resize_bicubic, melscale
        no_aug: bool = False,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        sa_portion: float = 0.15,
        sa_times: int = 2,
        fp16: bool = False,
        num_workers:int = 16, 
        pin_memory: bool = True,
        **kwargs):
    
    def nearest_multiple(x: int, target: int) -> int:
        assert (x > 0) and (target > 0) and (x < target)
        
        quotient = target // x
        lower_multiple = x * quotient
        upper_multiple = x * (quotient + 1)
        
        return lower_multiple if abs(target - lower_multiple) <= abs(target - upper_multiple) else upper_multiple


    if dataset == 'shd':
        sensor_size = tonic.datasets.SHD.sensor_size
    elif dataset == 'ssc':
        sensor_size = tonic.datasets.SSC.sensor_size
    else:
        raise
    
    
    #[Ch,F,T]
    print(f'dataset input_mothod="{input_method}"')
    if input_method == 'sum_bilinear':
        print(f'nearest_multiple(n_freq_bins={n_freq_bins}, sensor_size[0]={sensor_size[0]})={nearest_multiple(n_freq_bins, sensor_size[0])}')
        input_method_transform = [
            transforms.Resize(size=[nearest_multiple(n_freq_bins, sensor_size[0]), n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,F,T]->[Ch,T,F]
            transforms.Lambda(lambda x: x.reshape(1, n_time_bins, n_freq_bins, -1).sum(dim=-1, keepdim=False)),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,T,F]->[Ch,F,T]
        ]
    elif input_method == 'sum_bicubic':
        print(f'nearest_multiple(n_freq_bins={n_freq_bins}, sensor_size[0]={sensor_size[0]})={nearest_multiple(n_freq_bins, sensor_size[0])}')
        input_method_transform = [
            transforms.Resize(size=[nearest_multiple(n_freq_bins, sensor_size[0]), n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,F,T]->[Ch,T,F]
            transforms.Lambda(lambda x: x.reshape(1, n_time_bins, n_freq_bins, -1).sum(dim=-1, keepdim=False)),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,T,F]->[Ch,F,T]
        ]
    elif input_method == 'resize_bilinear':
        input_method_transform = [
            transforms.Resize(size=[n_freq_bins, n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)
        ]
    elif input_method == 'resize_bicubic':
        input_method_transform = [
            transforms.Resize(size=[n_freq_bins, n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC)
        ]
    elif input_method == 'melscale':
        input_method_transform = [
            transforms.Lambda(lambda x: x.flip(1).float()), #[Ch,F,T]... lauscher has high freq on lower F, but STFT has it on higher F
            torchaudio.transforms.MelScale(n_mels=n_freq_bins, 
                                           sample_rate=44000, 
                                           f_min=20, # higher than 20 -> low freq stretch=늘려서 edge로 더 가깝게
                                           f_max=20000, # lower than 20k -> high freq stretch=늘려서 edge로 더 가깝게
                                           n_stft=sensor_size[0]),
            transforms.Lambda(lambda x: x.flip(1)),                                           
        ]
    else:
        raise
    
    if  no_aug:
        train_transform = [
        # tonic 2 torch
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
        ] + input_method_transform 
    else:
        train_transform = [
            # tonic transforms
            tonic.transforms.DropEvent(p=drop_event),
            DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
            Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
            tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
            tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
            tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
        ] + [
            # tonic 2 torch
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
        ] + input_method_transform + [
            # torch transforms (specaug)
            torchaudio.transforms.FrequencyMasking(freq_mask_param=int(n_freq_bins*sa_portion), iid_masks=False),
            torchaudio.transforms.TimeMasking(time_mask_param=int(n_time_bins*sa_portion), iid_masks=False)
        ]*sa_times
    
    
    test_transform = [
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ] + input_method_transform
    
    
        
    if fp16:
        train_transform += [transforms.Lambda(lambda x: x.half())]
        test_transform += [transforms.Lambda(lambda x: x.half())]
        
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    print("train_transform", train_transform)
    print("test_transform", test_transform)
    

    
    if dataset == 'shd':
        train_dataset = tonic.datasets.SHD(dir, transform=train_transform, train=True)
        test_dataset = tonic.datasets.SHD(dir, transform=test_transform, train=False)
    elif dataset == 'ssc':
        train_dataset = tonic.datasets.SSC(dir, transform=train_transform, split='train')
        test_dataset = tonic.datasets.SSC(dir, transform=test_transform, split='test')
    else:
        raise


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=pin_memory, drop_last=True, num_workers=num_workers,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=pin_memory, drop_last=False, num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, test_loader

def get_spiking_audio_datasets_just_dataset(dataset:str, dir:str, batch_size:int, n_time_bins:int, n_freq_bins:int,
        input_method: str = 'sum_bilinear', #sum_bilinear, sum_bicubic, resize_bilinear, resize_bicubic, melscale
        no_aug: bool = False,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        sa_portion: float = 0.15,
        sa_times: int = 2,
        fp16: bool = False,
        num_workers:int = 16, 
        pin_memory: bool = True,
        **kwargs):
    
    def nearest_multiple(x: int, target: int) -> int:
        assert (x > 0) and (target > 0) and (x < target)
        
        quotient = target // x
        lower_multiple = x * quotient
        upper_multiple = x * (quotient + 1)
        
        return lower_multiple if abs(target - lower_multiple) <= abs(target - upper_multiple) else upper_multiple


    if dataset == 'shd':
        sensor_size = tonic.datasets.SHD.sensor_size
    elif dataset == 'ssc':
        sensor_size = tonic.datasets.SSC.sensor_size
    else:
        raise
    
    
    #[Ch,F,T]
    print(f'dataset input_mothod="{input_method}"')
    if input_method == 'sum_bilinear':
        print(f'nearest_multiple(n_freq_bins={n_freq_bins}, sensor_size[0]={sensor_size[0]})={nearest_multiple(n_freq_bins, sensor_size[0])}')
        input_method_transform = [
            transforms.Resize(size=[nearest_multiple(n_freq_bins, sensor_size[0]), n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,F,T]->[Ch,T,F]
            transforms.Lambda(lambda x: x.reshape(1, n_time_bins, n_freq_bins, -1).sum(dim=-1, keepdim=False)),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,T,F]->[Ch,F,T]
        ]
    elif input_method == 'sum_bicubic':
        print(f'nearest_multiple(n_freq_bins={n_freq_bins}, sensor_size[0]={sensor_size[0]})={nearest_multiple(n_freq_bins, sensor_size[0])}')
        input_method_transform = [
            transforms.Resize(size=[nearest_multiple(n_freq_bins, sensor_size[0]), n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,F,T]->[Ch,T,F]
            transforms.Lambda(lambda x: x.reshape(1, n_time_bins, n_freq_bins, -1).sum(dim=-1, keepdim=False)),
            transforms.Lambda(lambda x: x.permute(0,2,1)), #[Ch,T,F]->[Ch,F,T]
        ]
    elif input_method == 'resize_bilinear':
        input_method_transform = [
            transforms.Resize(size=[n_freq_bins, n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)
        ]
    elif input_method == 'resize_bicubic':
        input_method_transform = [
            transforms.Resize(size=[n_freq_bins, n_time_bins], 
                              interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC)
        ]
    elif input_method == 'melscale':
        input_method_transform = [
            transforms.Lambda(lambda x: x.flip(1).float()), #[Ch,F,T]... lauscher has high freq on lower F, but STFT has it on higher F
            torchaudio.transforms.MelScale(n_mels=n_freq_bins, 
                                           sample_rate=44000, 
                                           f_min=20, # higher than 20 -> low freq stretch=늘려서 edge로 더 가깝게
                                           f_max=20000, # lower than 20k -> high freq stretch=늘려서 edge로 더 가깝게
                                           n_stft=sensor_size[0]),
            transforms.Lambda(lambda x: x.flip(1)),                                           
        ]
    else:
        raise
    
    if  no_aug:
        train_transform = [
        # tonic 2 torch
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
        ] + input_method_transform 
    else:
        train_transform = [
            # tonic transforms
            tonic.transforms.DropEvent(p=drop_event),
            DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
            Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
            tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
            tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
            tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
        ] + [
            # tonic 2 torch
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
        ] + input_method_transform + [
            # torch transforms (specaug)
            torchaudio.transforms.FrequencyMasking(freq_mask_param=int(n_freq_bins*sa_portion), iid_masks=False),
            torchaudio.transforms.TimeMasking(time_mask_param=int(n_time_bins*sa_portion), iid_masks=False)
        ]*sa_times
    
    
    test_transform = [
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)), # [F,T,Ch]->[Ch,F,T]
    ] + input_method_transform
    
    
        
    if fp16:
        train_transform += [transforms.Lambda(lambda x: x.half())]
        test_transform += [transforms.Lambda(lambda x: x.half())]
        
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)
    print("train_transform", train_transform)
    print("test_transform", test_transform)
    

    
    if dataset == 'shd':
        train_dataset = tonic.datasets.SHD(dir, transform=train_transform, train=True)
        test_dataset = tonic.datasets.SHD(dir, transform=test_transform, train=False)
    elif dataset == 'ssc':
        train_dataset = tonic.datasets.SSC(dir, transform=train_transform, split='train')
        test_dataset = tonic.datasets.SSC(dir, transform=test_transform, split='test')
    else:
        raise

    return train_dataset, test_dataset