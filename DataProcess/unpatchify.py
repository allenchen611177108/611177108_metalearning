import os
import numpy as np
from PIL import Image
from patchify import unpatchify

def unpatch(path):
    img_dir = os.listdir(os.path.join(path,'pic'))
    mask_dir = os.listdir(os.path.join(path,'mask'))
    while (len(img_dir) >= 9):
        img_list = img_dir[:9]
        mask_list = mask_dir[:9]
        img_patch = np.zeros((3,3))
        mask_patch = np.zeros((3,3))
        
        count = 0
        for i in range(3):
            for j in range(3):
                img = Image.open(os.path.join(path,'pic',img_list[count]))
                img = np.asarray(img)
                img_patch[i][j] = img


unpatch("C:/allen_env/deeplearning/7f/fold_4/test_set")