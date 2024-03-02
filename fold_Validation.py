import os
import torch
import torch.nn as nn
from MLFunc.Model import FCN 
from torchvision import transforms
from MLFunc.dataset import reductionset
# from PIL import Image
import itertools
from torchmetrics import JaccardIndex

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

batch_size = 1
bias = 1e-5

mask_trans = transforms.Compose(
    [transforms.ToTensor()]
)
img_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

randset = reductionset('C:/allen_env/deeplearning/7f/fold_5/train_set', transform=img_trans,
                   mask_transform=mask_trans)
randloader = torch.utils.data.DataLoader(randset, batch_size=1, shuffle=True, num_workers=0)

validset = reductionset('C:/allen_env/deeplearning/7f/fold_5/test_set', transform=img_trans,
                   mask_transform=mask_trans)
vertifyloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, num_workers=0)

# def matplot_generate(path_list, output):
#     i = 0
#     for ref in path_list:
#         image = Image.open(ref).convert("RGB")
#         image.save("C:/Users/cclinlab/Desktop/實驗結果/fold7/test/{}_a.png".format(i))
#         image.show()
#         mask_tensor = output[i]
#         transform = transforms.ToPILImage()
#         mask = transform(mask_tensor)
#         mask.save("C:/Users/cclinlab/Desktop/實驗結果/fold7/test/{}_b.png".format(i))
#         mask.show()
#         i += 1

jaccard = JaccardIndex(task='multiclass', num_classes=2)
def IoU(epoch=1, sample_num=10, dataloader=None, model=None):
    max_IoU = 0.0
    for i in range(epoch):
        total_IOU = 0.0
        for img, mask in itertools.islice(dataloader, 0, sample_num):
            img, mask = img.cuda(), mask.cuda()
            predict = model(img)

            predict = predict.cpu()

            sigmoid = nn.Sigmoid()
            predict1 = sigmoid(predict)
            threshold = torch.tensor([0.5])

            mask = mask[0][0].cpu()

            # 實測後，[0][1] 最能表達理想結果
            predict1_1 = (predict1[0][1] > threshold).float() * 1
            iou = jaccard(predict1_1, mask)
            if torch.isnan(iou).any():
                iou_current = 0.0
            else:
                iou_current = iou.item()
            total_IOU = total_IOU + iou_current
            torch.cuda.empty_cache()
        avg_iou = float(total_IOU / sample_num)
        if (avg_iou > max_IoU): max_IoU = avg_iou
    return max_IoU


model_path = "C:/allen_env/deeplearning/result/model/FoMAMLFCN_7fold5.pth"
test_net = FCN(2)
test_net.load_state_dict(torch.load(model_path)['param'])
# test_net = torch.load(model_path) #舊模型

test_net.cuda()

test_net.eval()
total_IOU = 0.0
length = 500

try:
    train_IoU = IoU(epoch=10, sample_num=length, dataloader=randloader, model=test_net)
    print("average_Train_iou = %f.\n" %train_IoU)
    valid_IoU = IoU(epoch=10, sample_num=length, dataloader=vertifyloader, model=test_net)
    print("average_Valid_iou = %f.\n" %valid_IoU)
except Exception as err:
    print(err)

torch.cuda.empty_cache()
# matplot_generate(randset.path, trainoutput)
# matplot_generate(validset.path, testoutput)