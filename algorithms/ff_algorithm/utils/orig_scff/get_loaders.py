import torch
import torchvision
from torchvision.transforms import transforms, ToPILImage, Compose, ToTensor, RandomAffine, Lambda
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, SVHN
import user_variables as uv

# Define the custom CIFAR-10 dataset

#custom the trainloader to include the augmented views of the original batch
torch.manual_seed(1234)
# Define the two sets of transformations
#BATCHSIZE = 50
s = 0.5
transform1 = transforms.Compose([
    #transforms.RandomCrop(32, padding=0),
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # using default scale range
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(uv.dtype),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform2 = transforms.Compose([
    #transforms.RandomCrop(32, padding=0),
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # using default scale range
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(uv.dtype),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding = 1),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(uv.dtype),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#transform_test = transform_train

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(uv.dtype),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class DualAugmentCIFAR10(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR-10 dataset that applies dual augmentation techniques 
    for unsupervised SCFF. (default: no augmentation is used) 

    Args:
        root (str): Root directory where the dataset is stored.
        augment (str): Type of augmentation to apply. 
                       Options: 'no' (default), 'single', 'dual'.
        *args: Additional arguments for the CIFAR-10 dataset.

    Attributes:
        augment (str): Stores the selected augmentation mode.
    """
    def __init__(self, root, augment="No", *args, **kwargs):
        super(DualAugmentCIFAR10, self).__init__(root,*args, **kwargs)
        self.augment = augment
        
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img_pil = ToPILImage()(img)
        img_original = transform_train(img_pil)

        if self.augment == "single":
            img1 = transform1(img_pil)
            return img_original, img1, img_original, target
        elif self.augment == "dual":
            img1 = transform1(img_pil)
            img2 = transform2(img_pil)
            return img_original, img1, img2, target
        else:
            return img_original, target

            
class DualAugmentCIFAR10_test(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR-10 dataset that applies augmentation techniques 
    for supervised evaluation of the trained model with SCFF.

    Args:
        aug (bool): Whether to apply data augmentation to test images.
        *args: Additional arguments for the CIFAR-10 dataset.

    Attributes:
        aug (bool): Stores whether augmentation is applied. True for train set, False for test set
    """
    def __init__(self, aug=False, *args, **kwargs):
        super(DualAugmentCIFAR10_test, self).__init__(*args, **kwargs)
        self.aug = aug
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = ToPILImage()(img)
        
        if self.aug:
            img = transform_train(img)
        else:
            img = transform_test(img)
        
        return img, target
    
def get_train_cifar(batchsize, augment):
    """
    Creates data loaders for CIFAR-10 training and validation.

    Args:
        batchsize (int): Batch size for training.
        augment (str): Data augmentation strategy (e.g., 'no', 'single', 'dual').
        factor (float): Proportion of dataset to use for training.

    Returns:
        tuple: (train_loader, val_loader, test_loader, sup_train_loader)
    """
    torch.manual_seed(1234)
    trainset = DualAugmentCIFAR10(root='./data', train=True, download=True, augment=augment)
    sup_trainset = DualAugmentCIFAR10_test(root='./data', aug = True, train=True, download=True)
    train_len = int(len(trainset))
    #val_len = len(trainset) - train_len

    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_len]
    # val_indices = indices[train_len:]

    # Create subsets
    train_data = Subset(trainset, train_indices)
    sup_train_data = Subset(sup_trainset, train_indices)
    # val_data = Subset(sup_trainset, val_indices)

    testset = DualAugmentCIFAR10_test(root='./data',aug = False, train=False, download=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    #train_data, val_data = random_split(trainset, [train_len, val_len])

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
    valloader = testloader

    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True, )

    return trainloader, valloader, testloader, sup_trainloader



class AugmentedMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super(AugmentedMNIST, self).__init__(root, train=train, transform=transform, 
                                             target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # Convert image to PIL Image for transformation
        img = ToPILImage()(img)

        # Apply the original transform
        if self.transform is not None:
            orig_img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return orig_img, target # , aug_img_1, aug_img_2, target


class CustomMNIST(MNIST):
    def __init__(self, root, train=True, transform=None,download=False):
        super(CustomMNIST, self).__init__(root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # Convert image to PIL Image for transformation
        img = ToPILImage()(img)

        # Apply the original transform
        if self.transform is not None:
            orig_img = self.transform(img)

        return orig_img, target


def get_train_mnist(batchsize):

    torch.manual_seed(42)
    # Transformation pipeline
    # transform = Compose([
    #     ToTensor(),
    #     Lambda(lambda x: torch.flatten(x))])
    transform = Compose([
        ToTensor(),
        transforms.ConvertImageDtype(uv.dtype)])

    trainset = AugmentedMNIST(root='data', train=True, download=True, transform=transform)
    #mnist_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_tr)
    mnist_test = torchvision.datasets.MNIST(root='data', download=True, train=False, transform=transform)

    sup_trainset = CustomMNIST(root='data',transform=transform, train=True, download=True)

    train_size = 60000
    val_size = 0

    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]

    # Create subsets
    mnist_train = Subset(trainset, train_indices)
    sup_train_data = Subset(sup_trainset, train_indices)
    mnist_val = Subset(trainset, val_indices)

    train_loader = DataLoader(mnist_train, batch_size= batchsize, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size= batchsize, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size= 1000, shuffle=False)
    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader, sup_trainloader

def get_train_svhn(batchsize):

    torch.manual_seed(42)

    # SVHN: [0,255] -> [0,1] -> float16
    transform = Compose([
        ToTensor(),
        transforms.ConvertImageDtype(uv.dtype),
    ])

    # SVHN에서는 label 10이 '0' 숫자를 의미해서 보통 0으로 매핑해서 씀
    target_transform = lambda y: 0 if y == 10 else y

    # train / test dataset
    trainset = SVHN(
        root='data',
        split='train',
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    svhn_test = SVHN(
        root='data',
        split='test',
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    # sup_trainset: 필요하면 나중에 CustomSVHN으로 교체 가능
    sup_trainset = SVHN(
        root='data',
        split='train',
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    # train/val split (지금은 val_size = 0으로 MNIST 코드와 동일하게)
    train_size = len(trainset)
    val_size = 0

    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]

    svhn_train = Subset(trainset, train_indices)
    svhn_val = Subset(trainset, val_indices)
    sup_train_data = Subset(sup_trainset, train_indices)

    train_loader = DataLoader(svhn_train, batch_size=batchsize,
                              shuffle=True)
    val_loader = DataLoader(svhn_val, batch_size=batchsize,
                            shuffle=False)
    test_loader = DataLoader(svhn_test, batch_size=1000,
                             shuffle=False)
    sup_trainloader = DataLoader(sup_train_data, batch_size=64,
                                 shuffle=True)

    return train_loader, val_loader, test_loader, sup_trainloader