import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from MLFunc.dataset import dataset
import MLFunc.Extra as ex_func
from MLFunc.Model import FCN
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 1
bias = 1e-5

mask_trans = transforms.Compose(
    [transforms.ToTensor()]
)
img_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = dataset('C:/allen_env/deeplearning/7f/fold_5/train_set', transform=img_trans,
                   mask_transform=mask_trans)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

model_path = "C:/allen_env/deeplearning/result/Meta/ReptileFCN.pth"
model = FCN(2)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.85)

target_epoch = 675

# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=80, min_lr=0.000001, cooldown=20)

breakpoint = {
    "param": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": 0
}
best_error = 1.
best_epoch = 0

model.train()

fields = ['Epoch', 'avg_error', 'Lr'] 
record = []

start = time.time()
try:
    for epoch in range(target_epoch):
        error = 0.
        for i, data in enumerate(trainloader, 0):
            img, mask = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask[0].long()).cpu()
            error = error + loss
            loss.cpu().backward()
            optimizer.step()
            torch.cuda.empty_cache()
        scheduler.step()
        error = error / trainset.__len__()
        if(error < best_error):
            best_error = error
            best_epoch = epoch
            breakpoint['model'] = model.state_dict()
            breakpoint["optimizer"] = optimizer.state_dict()
            breakpoint["epoch"] = best_epoch
        lr = ex_func.get_lr(optimizer=optimizer)
        print('Epoch: %s | avg_Err: %s | Lr: %s' %(str(epoch), str(error.item()), str(lr)))
        item = []
        item.append(str(epoch))
        item.append(str(error.item()))
        item.append(str(lr))
        record.append(item)
    torch.save(breakpoint, 'C:/allen_env/deeplearning/result/checkpoint/ReptileFCN_7fold5_bp_best_%s.pth' %str(best_epoch))
    Model = {'param': model.state_dict()}
    torch.save(Model, 'C:/allen_env/deeplearning/result/model/ReptileFCN_7fold5.pth')
except Exception as err:
    print(err)
    torch.save(breakpoint, 'C:/allen_env/deeplearning/result/checkpoint/ReptileFCN_7fold5_bp_best_%s.pth' %str(best_epoch))

torch.cuda.empty_cache()

end = time.time()
print("執行時間：%f 秒" % (end - start))

with open('ReptileFCN_7fold5', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(fields)
    write.writerows(record)
