from os.path import basename
import glob
import math
import numpy as np
import cv2

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * np.log10(255.0 / np.sqrt(mse))

GT_path = 'Water_Net/'
img_path = 'uwcnn/'
img_list = glob.glob('uwcnn/prediction_v1/*.png')

total_ssim, total_psnr = 0, 0

for i in range(len(img_list)):

    #img1 = cv2.imread(img_path + 'input/' + basename(img_list[i]))
    #img2 = cv2.imread(img_path + 'output/'+ basename(img_list[i]).split('.')[0] + '.png_out.png')
    
    print('img_name : \t', basename(img_list[i]))

    img1 = cv2.imread(GT_path + 'GT/' + basename(img_list[i]))
    img2 = cv2.imread(img_path + 'prediction_v1/'+ basename(img_list[i]))
    
    img1_h, img1_w, _ = img1.shape
    img2_h, img2_w, _ = img2.shape

    print('img1.shape : \t', img1.shape)
    print('img2.shape : \t', img2.shape)

    if img2_h > img1_h: 
        img2 = img2[0:img1_h, :, :]
    elif img2_w > img1_w:
        img2 = img2[:, 0:img1_w, :]
    elif img1_h > img2_h:
        img1 = img1[0:img2_h, :, :]
    elif img1_w > img2_w:    
        img1 = img1[:, 0:img2_w, :]

    img1_h, img1_w, _ = img1.shape
    img2_h, img2_w, _ = img2.shape

    print('img1.shape : \t', img1.shape)
    print('img2.shape : \t', img2.shape)

    if img2_h > img1_h: 
        img2 = img2[0:img1_h, :, :]
    elif img2_w > img1_w:
        img2 = img2[:, 0:img1_w, :]
    elif img1_h > img2_h:
        img1 = img1[0:img2_h, :, :]
    elif img1_w > img2_w:    
        img1 = img1[:, 0:img2_w, :]
    

    #else: 
    #    img1 = img1[0:img2_h, 0:img2_w, :]

    # cv2.imwrite(img_path + 'testing_cropped/' + 'in_' + basename(img_list[i]), img1)
    # cv2.imwrite(img_path + 'testing_cropped/' + 'op_' +  basename(img_list[i]), img2)
    '''
    print('img1.shape : \t', img1.shape)
    print('img2.shape : \t', img2.shape)

    print('img1 name : \t', basename(img_list[i]))
    print('img2 name : \t', basename(img_list[i]))
    '''
    total_psnr = total_psnr + calculate_psnr(img1, img2)
    total_ssim = total_ssim + calculate_ssim(img1, img2)


    print('PSNR : \t', calculate_psnr(img1, img2))
    print('SSIM : \t', calculate_ssim(img1, img2))

print('Len img_list : \t', len(img_list))
print('Avg. PSNR : \t', total_psnr/len(img_list))
print('Avg. SSIM : \t', total_ssim/len(img_list))
