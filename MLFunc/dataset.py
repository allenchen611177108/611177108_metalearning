import random
import torch.utils
import os
from PIL import Image
from torchvision import transforms

def reductfilter(path):
    tf = transforms.Compose([transforms.PILToTensor()])
    graph = Image.open(path).convert("L")
    graph = tf(graph)
    scale = graph.size(1) * graph.size(2)
    num_white = (graph == 255).sum()
    prop = num_white / scale
    if(prop > 0.85):
        return False
    else:
        return True

class dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mask_transform=None):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        data_name = self.imgs[idx]
        img_path = os.path.join(self.root, "pic", data_name)
        mask_path = os.path.join(self.root, "mask", data_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)
    
class reductionset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mask_transform=None):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        while(True):
            idx = random.randint(0, len(self.imgs) - 1)
            data_name = self.imgs[idx]
            img_path = os.path.join(self.root, "pic", data_name)
            if(reductfilter(img_path) == True):
                break
            else:
                continue
        mask_path = os.path.join(self.root, "mask", data_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)

class rand_dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mask_transform=None):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.transform = transform
        self.mask_transform = mask_transform
        self.path = []
    def __getitem__(self, idx):
        idx = random.randint(0, len(self.imgs) - 1)
        data_name = self.imgs[idx]
        img_path = os.path.join(self.root, "pic", data_name)
        self.path.append(img_path)
        mask_path = os.path.join(self.root, "mask", data_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)
# print(reductfilter("C:/allen_env/deeplearning/7f/fold_1/train_set/pic/0_12.png"))