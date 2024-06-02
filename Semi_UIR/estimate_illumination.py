import cv2
import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join

# estimate the illumination map

def luminance_estimation(img):
    sigma_list = [15, 60, 90]
    img = np.uint8(np.array(img))
    illuminance = np.ones_like(img).astype(np.float32)
    for sigma in sigma_list:
        illuminance1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-8)
        illuminance1 = np.clip(illuminance1, 0, 255)
        illuminance = illuminance + illuminance1
    illuminance = illuminance / 3
    L = (illuminance - np.min(illuminance)) / (np.max(illuminance) - np.min(illuminance) + 1e-6)
    L = np.uint8(L * 255)
    return L


input_dir = "D:/UnderWaterResearch/Codes/Semi-UIR/data/test/Seathru2K_D2/input"
input_lists = glob(join(input_dir, "*.*"))
result_dir = "D:/UnderWaterResearch/Codes/Semi-UIR/data/test/Seathru2K_D2/LA/"

# print('input_lists : \t', input_lists)

for gen_path in zip(input_lists):
    print('gen_path[0] : \t', gen_path[0])

    img = Image.open(gen_path[0])
    print('img.shape : \t', img)

    img_name = gen_path[0].split('\\')[1]
    
    print('img_name : \t', img_name)
    print('path_join : \t', os.path.join(result_dir, img_name))
    
    L = luminance_estimation(img)
    ndar = Image.fromarray(L)
    ndar.save(os.path.join(result_dir, img_name))

print('finished!')
