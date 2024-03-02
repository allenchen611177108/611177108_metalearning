import os
import time
import random
import torch
import torch.nn as nn
from MLFunc.Model import FCN
import MLFunc.Extra as ex_func
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as et
import xml.dom.minidom

# hyper parameter
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
inner_lr = 0.01
lr = 0.05
meta_epoch = 50

start = "C:/allen_env\deeplearning\metaDataset"
dir = []
for dirlv1 in (os.listdir(start)):
    node1 = os.path.join(start, dirlv1)
    for dirlv2 in (os.listdir(node1)):
        node2 = os.path.join(node1, dirlv2)
        dir.append(node2)
order = [8, 0, 1, 2, 4, 5, 6, 3, 7] #learn low scale image first, then learn fragment and vessel in different color, finally learn the dark color feature
dir = [dir[i] for i in order]

img_transform = transforms.Compose([transforms.ToTensor()])
mask_transform = transforms.Compose([transforms.ToTensor()])

class dataset(Dataset):
    def __init__(self, root, img_trans = None, mask_trans = None):
        self.dir_root = root
        self.img_path = os.path.join(self.dir_root, 'pic')
        self.mask_path = os.path.join(self.dir_root, 'mask')
        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)
        self.img_trans = img_trans
        self.mask_trans = mask_trans
    def __getitem__(self, index):
        img_name = self.img_list[index]
        file_path = os.path.join(self.img_path, img_name)
        img = Image.open(file_path).convert('RGB')
        if self.img_trans is not None:
            img = self.img_trans(img)
        img = torch.unsqueeze(img, dim=0) # neural network`s input is not a image, is a image in the "batch"
        mask_name = self.mask_list[index]
        maskfile_path = os.path.join(self.mask_path, mask_name)
        mask = Image.open(maskfile_path).convert('L')
        if self.mask_trans is not None:
            mask = self.mask_trans(mask)
        return img, mask
    def __len__(self):
        return len(self.img_list)

model = FCN(num_classes=2)
model.cuda()
model_copy = FCN(num_classes=2)
model_copy.load_state_dict(model.state_dict())
model_copy.cuda()

criterion = nn.CrossEntropyLoss()
meta_optimizer = optim.SGD(model_copy.parameters(), lr=inner_lr, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

model_copy.train()
model.train()
torch.autograd.set_detect_anomaly(True)

start = time.time()

Root = et.Element("Record")

for i in range(meta_epoch):
    print("Meta Epoch: %d \n" %i)
    tag_epoch = "Epoch_%d" %i
    epoch_record = et.Element(tag_epoch)
    meta_optimizer.zero_grad()
    meta_loss = 0
    for site in dir:
        # every tasks
        d = dataset(os.path.join(site, 'train'), img_trans=img_transform, mask_trans=mask_transform)
        d_test = dataset(os.path.join(site, 'test'), img_trans=img_transform, mask_trans=mask_transform)
        seq_list = random.sample(range(0, d.__len__()), 6)
        for idx in seq_list:
            meta_optimizer.zero_grad()
            img, mask = d[idx]
            output = model_copy(img.cuda())
            loss = criterion(output, mask.long().cuda())
            loss.cpu()
            loss.backward()
            meta_optimizer.step()
            meta_lr = ex_func.get_lr(optimizer=meta_optimizer)

            E_loss = et.Element("loss")
            E_loss.text = str(loss.item())
            m_lr = et.Element("meta_lr")
            m_lr.text = str(meta_lr)

            torch.cuda.empty_cache()
            
        idx_test = random.randint(0, d_test.__len__() - 1)
        img_test, mask_test = d_test[idx_test]
        output_test = model_copy(img_test.cuda())
        task_loss = criterion(output_test, mask_test.long().cuda())
        task_loss.cpu()
        task_loss = task_loss.detach()
        meta_loss = meta_loss + task_loss

        t_loss = et.Element("task_loss")
        t_loss.text = str(task_loss.item())

        torch.cuda.empty_cache()

    optimizer.zero_grad()
    meta_loss = Variable(meta_loss, requires_grad = True)
    meta_loss.backward()
    optimizer.step()
    main_lr = ex_func.get_lr(optimizer=optimizer)

    E_metaloss = et.Element("meta_loss")
    E_metaloss.text = str(meta_loss.item())
    epoch_record.append(E_metaloss)
    E_lr = et.Element("lr")
    E_lr.text = str(main_lr)
    epoch_record.append(E_lr)
    Root.append(epoch_record)

    torch.cuda.empty_cache()

torch.save({
            'model_state_dict':model.state_dict(),
            }, 'C:/allen_env/deeplearning/result/model/FCN_meta_1c_v2.pth')

end = time.time()
print("執行時間：%f 秒" % (end - start))

xml_string = et.tostring(Root)
dom = xml.dom.minidom.parseString(xml_string)
pretty_xml = dom.toprettyxml()

with open("C:/allen_env/deeplearning/code/MetaLearning/record/meta_1c.xml", "w") as file:
    file.write(pretty_xml)