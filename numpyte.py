import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from tqdm import trange
import time

def EDI_downscale(img,scale):
    
    w, h, wd = img.shape
    scale = int(scale)
    imgo = np.zeros((w//scale, h//scale, 3))


    for i in trange(w//scale):
        for j in range(h//scale):
            for rgb in range(0, wd):
                avg = img[i * scale: i * scale + scale, j * scale : j * scale + scale,(rgb)]
                imgo[i][j][rgb] = np.mean(avg)
            
    return imgo.astype(img.dtype)

def EDI_upscale(img, m):
    
    # m should be equal to a power of 2
    if m%2 != 0:
        m += 1
        
    # initializing image to be predicted
    w, h,wd = img.shape
    imgo = cv2.resize(img,(h * 2, w * 2),interpolation=cv2.INTER_LINEAR)
    # print(imgo.shape)
    # imgo = np.zeros((w*2,h*2,wd))
    # print(imgo.shape)
    # input()
    
    # Place low-resolution pixels
    for i in range(w):
        for j in range(h):
            imgo[2*i][2*j] = img[i][j]    

    y = np.zeros((m**2,1)) # pixels in the window
    C = np.zeros((m**2,4)) # interpolation neighbours of each pixel in the window
    
    # Reconstruct the points with the form of (2*i+1,2*j+1)
    for w in range(0,wd):
        for i in range(math.floor(m/2), w-math.floor(m/2)):
            for j in range(math.floor(m/2), h-math.floor(m/2)):
                tmp = 0
                for ii in range(i-math.floor(m/2), i+math.floor(m/2)):
                    for jj in range(j-math.floor(m/2), j+math.floor(m/2)):
                        y[tmp][0] = imgo[2*ii][2*jj][w]
                        C[tmp][0] = imgo[2*ii-2][2*jj-2][w]
                        C[tmp][1] = imgo[2*ii+2][2*jj-2][w]
                        C[tmp][2] = imgo[2*ii+2][2*jj+2][w]
                        C[tmp][3] = imgo[2*ii-2][2*jj+2][w]
                        tmp += 1

                # calculating weights
                # a = (C^T * C)^(-1) * (C^T * y) = (C^T * C) \ (C^T * y)
                a = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(C),C)), np.transpose(C)), y)
                imgo[2*i+1][2*j+1][w] = np.matmul([imgo[2*i][2*j][w], imgo[2*i+2][2*j][w], imgo[2*i+2][2*j+2][w], imgo[2*i][2*j+2]][w], a)
                
        # Reconstructed the points with the forms of (2*i+1,2*j) and (2*i,2*j+1)
        for i in range(math.floor(m/2), w-math.floor(m/2)):
            for j in range(math.floor(m/2), h-math.floor(m/2)):
                tmp = 0
                for ii in range(i-math.floor(m/2), i+math.floor(m/2)):
                    for jj in range(j-math.floor(m/2), j+math.floor(m/2)):
                        y[tmp][0] = imgo[2*ii+1][2*jj-1][w]
                        C[tmp][0] = imgo[2*ii-1][2*jj-1][w]
                        C[tmp][1] = imgo[2*ii+1][2*jj-3][w]
                        C[tmp][2] = imgo[2*ii+3][2*jj-1][w]
                        C[tmp][3] = imgo[2*ii+1][2*jj+1][w]
                        tmp += 1

                # calculating weights
                # a = (C^T * C)^(-1) * (C^T * y) = (C^T * C) \ (C^T * y)
                a = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(C),C)), np.transpose(C)), y)
                imgo[2*i+1][2*j] = np.matmul([imgo[2*i][2*j][w], imgo[2*i+1][2*j-1][w], imgo[2*i+2][2*j][w], imgo[2*i+1][2*j+1]][w], a)
                imgo[2*i][2*j+1] = np.matmul([imgo[2*i-1][2*j+1][w], imgo[2*i][2*j][w], imgo[2*i+1][2*j+1][w], imgo[2*i][2*j+2]][w], a)
        
    # Fill the rest with bilinear interpolation
    np.clip(imgo, 0, 255.0, out=imgo)
    # imgo_bilinear = cv2.resize(img, dsize=(h*2,w*2), interpolation=cv2.INTER_LINEAR)
    # imgo[imgo==[0,0,0]] = imgo_bilinear[imgo==[0,0,0]]
    
    return imgo.astype(img.dtype)

def EDI_predict(img, m, s):

    try:
        w, h,_ = img.shape
    except:
        sys.exit("Error input: Please input a valid grayscale image!")
    
    output_type = img.dtype

    if s <= 0:
        sys.exit("Error input: Please input s > 0!")
        
    elif s == 1:
        print("No need to rescale since s = 1")
        return img
    
    elif s < 1:
        # Calculate how many times to do the EDI downscaling
        n = math.floor(math.log(1/s, 2))
        
        # Downscale to the expected size with linear interpolation
        linear_factor = 1/s / math.pow(2, n)
        if linear_factor != 1:
            img = cv2.resize(img, dsize=(int(h/linear_factor),int(w/linear_factor)), interpolation=cv2.INTER_LINEAR).astype(output_type)

        for i in range(n):
            img = EDI_downscale(img)
        return img
        
    elif s < 2:
        # Linear Interpolation is enough for upscaling not over 2
        return cv2.resize(img, dsize=(int(h*s),int(w*s)), interpolation=cv2.INTER_LINEAR).astype(output_type)
    
    else:
        # Calculate how many times to do the EDI upscaling
        n = math.floor(math.log(s, 2))
        for i in range(n):
            img = EDI_upscale(img, m)
        
        # Upscale to the expected size with linear interpolation
        linear_factor = s / math.pow(2, n)
        if linear_factor == 1:
            return img.astype(output_type)

        # Update new shape
        w, h = img.shape
        return cv2.resize(img, dsize=(int(h*linear_factor),int(w*linear_factor)), interpolation=cv2.INTER_LINEAR).astype(output_type)

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    print("读取图片" + sys.argv[1] + "完成")
    img = EDI_predict(img,4,4)
    print("nedi超采样完成")
    w,h,_ = img.shape
    img = cv2.resize(img,(h * 2, w * 2),interpolation=cv2.INTER_LINEAR)
    print("cubic抗锯齿超采样完成")
    cv2.imwrite(sys.argv[2],img)
    print("保存图片到:" + sys.argv[2])
    

