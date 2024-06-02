from hmac import new
from itertools import count
from os.path import basename
import glob
import math
import shutil
import numpy as np
import cv2


img_list = glob.glob('uwcnn/prediction/*.png')
dest_folder = 'uwcnn/'
total_ssim, total_psnr = 0, 0
count = 0

for i in range(len(img_list)):

    img_name = basename(img_list[i]).split('.')[1]
    print('before img_name : \t', basename(img_list[i]))
    if img_name == 'png_out':
        print('img_name : \t ', basename(img_list[i]))
        new_name = basename(img_list[i]).split('.')[0]
        shutil.copy(img_list[i], dest_folder+new_name+'.png')
        count += 1
        # print('img_name : \t ', img_name)

    else:
        print('img_name : \t ', basename(img_list[i]))
        continue


print('count : \t', count)