from datasets.utils import SeededDataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from globals import CONFIG

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    transform.append(T.Resize((size, size)))
    if preprocess:
        transform.append(T.RandomCrop(size=size, padding=padding))
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean, std))
    return T.Compose(transform)

def load_data():
    size = 32
    if 'pretrain' in CONFIG.experiment_args:
        size = 224

    CONFIG.num_classes = 10
    CONFIG.data_input_size = (3, size, size)

    mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
    train_transform = get_transform(size=size, padding=4, mean=mean, std=std, preprocess=True)
    test_transform = get_transform(size=size, padding=4, mean=mean, std=std, preprocess=False)
    train_dataset = CIFAR10(CONFIG.dataset_args['root'], train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(CONFIG.dataset_args['root'], train=False, download=True, transform=test_transform)

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
