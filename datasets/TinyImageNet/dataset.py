import io
import pandas as pd
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
from datasets.utils import SeededDataLoader
import torchvision.transforms as T

from torchvision import datasets

from globals import CONFIG

# Based on https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
def TINYIMAGENET(root, train=True, transform=None, target_transform=None, download=False):
    
    def _exists(root, filename):
        return os.path.exists(os.path.join(root, filename))

    def _download(url, root, filename):
        datasets.utils.download_and_extract_archive(url=url, 
                                                    download_root=root, 
                                                    extract_root=root, 
                                                    filename=filename)

    def _setup(root, base_folder):
        target_folder = os.path.join(root, base_folder, 'val/')

        val_dict = {}
        with open(target_folder + 'val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
                
        paths = glob.glob(target_folder + 'images/*')
        paths[0].split('/')[-1]
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(target_folder + str(folder)):
                os.mkdir(target_folder + str(folder))
                
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            dest = target_folder + str(folder) + '/' + str(file)
            move(path, dest)
            
        os.remove(target_folder + 'val_annotations.txt')
        rmdir(target_folder + 'images')

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'

    if download and not _exists(root, filename):
        _download(url, root, filename)
        _setup(root, base_folder)
    folder = os.path.join(root, base_folder, 'train' if train else 'val')

    return datasets.ImageFolder(folder, transform=transform, target_transform=target_transform)

def get_transform(size, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(T.RandomResizedCrop(size=size, scale=(0.1, 1.0), ratio=(0.8, 1.25)))
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean, std))
    return T.Compose(transform)

def load_data():
    size = 64
    if 'pretrain' in CONFIG.experiment_args:
        size = 224

    CONFIG.num_classes = 200
    CONFIG.data_input_size = (3, size, size)

    mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
    train_transform = get_transform(size=size, mean=mean, std=std, preprocess=True)
    test_transform = get_transform(size=size, mean=mean, std=std, preprocess=False)
    train_dataset = TINYIMAGENET(CONFIG.dataset_args['root'], train=True, download=True, transform=train_transform)
    test_dataset = TINYIMAGENET(CONFIG.dataset_args['root'], train=False, download=True, transform=test_transform)

    train_loader = SeededDataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = SeededDataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return {'train': train_loader, 'test': test_loader}
