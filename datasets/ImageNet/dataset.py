import os
from datasets.utils import SeededDataLoader
import torchvision.transforms as T

from torchvision import datasets

from globals import CONFIG

def get_transform(train, mean, std):
    if train:
        transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.2,1.)),
            T.RandomGrayscale(p=0.2),
            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)])
    else:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std)])
    return transform

def load_data():
    CONFIG.num_classes = 1000
    CONFIG.data_input_size = (3, 224, 224)

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = get_transform(train=True, mean=mean, std=std)
    test_transform = get_transform(train=False, mean=mean, std=std)

    train_folder = os.path.join(CONFIG.dataset_args['root'], 'train')
    train_dataset = datasets.ImageFolder(train_folder, transform=train_transform)
    test_folder = os.path.join(CONFIG.dataset_args['root'], 'val')
    test_dataset = datasets.ImageFolder(test_folder, transform=test_transform)

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
