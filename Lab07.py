import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack
from scipy.interpolate import interp1d
import sys
import math 
from math import sqrt
from datetime import date
from datetime import datetime
import cv2 as cv2
from tqdm import tqdm

import os
cls = lambda: os.system('cls')
cls()

img = cv2.imread('test12.png', cv2.IMREAD_COLOR) # format -> BGR

h = img.shape[0]
w = img.shape[1]

print("\n img = \n\n", img)
print("\n img.dtype = ", img.dtype)
print("\n type(img) = ", type(img))
print("\n img.shape = ", img.shape)

RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("\n RGB = \n\n", RGB)
print("\n RGB.dtype = ", RGB.dtype)
print("\n type(RGB) = ", type(RGB))
print("\n RGB.shape = ", RGB.shape)

plt.imshow(RGB)
plt.show()

###################################################################

YCrCb = cv2.cvtColor(RGB, cv2.COLOR_RGB2YCrCb).astype(int)

print("\n YCrCb = \n\n", YCrCb)
print("\n YCrCb.dtype = ", YCrCb.dtype)
print("\n type(YCrCb) = ", type(YCrCb))
print("\n YCrCb.shape = ", YCrCb.shape)

"""    
    y = np.zeros((h, w), np.uint8) + img[:, :, 0]
    cr = np.zeros((h, w), np.uint8) + img[:, :, 1]
    cb = np.zeros((h, w), np.uint8) + img[:, :, 2]       

    print("\n y.dtype = ", y.dtype)
    print("\n cr.dtype = ", cr.dtype)
    print("\n cb.dtype = ", cb.dtype)
    
    y = y - 128
    cr = cr - 128
    cb = cb - 128

    #print("\n y = ", y)
    #print("\n cr = ", cr)
    #print("\n cb = ", cb)

"""

###################################################################

YCrCb[:,:,0] = YCrCb[:,:,0] - 128
YCrCb[:,:,1] = YCrCb[:,:,1] - 128
YCrCb[:,:,2] = YCrCb[:,:,2] - 128

###################################################################

def chroma_subsampling(YCrCb, type):
   
    if(type == "4:2:2"):

        return YCrCb

    else: 

        for w in range(2):
            for i in range(YCrCb.shape[0]):
                for j in range(YCrCb.shape[1]):

                    if((j % 2)!=0): 
                    
                        if(w == 0):

                            YCrCb[i,j,1] = YCrCb[i,j-1,1]

                        if(w == 1):
                            
                            YCrCb[i,j,2] = YCrCb[i,j-1,2]

    return YCrCb

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def quantization(dct_block, Qy, Qc, k, quant_table): 
    
    if(quant_table == False):

        return dct_block.astype(int)

    else:

        if(k == 0):

            #dct_block = np.round(dct_block/Qy).astype(int)
            dct_block = (dct_block/Qy).astype(int)
        else:
            #dct_block = np.round(dct_block/Qc).astype(int)
            dct_block = (dct_block/Qc).astype(int)      

        return dct_block


def dequantization(dct_block, Qy, Qc, k, quant_table):    

    if(quant_table == False):

        return dct_block
    else:

        if(k == 0):

            #dct_block = np.round(dct_block/Qy).astype(int)
            dct_block = (dct_block*Qy)
        else:
            #dct_block = np.round(dct_block/Qc).astype(int)
            dct_block = (dct_block*Qc)       

        return dct_block

def zigzag(A): 

    template= n= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])

    if len(A.shape)==1:

        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]

    else:

        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):                
                B[template[r,c]]=A[r,c] 
    return B

def TwoD(a, b):    
    
        lst = [['0' for col in range(a)] for col in range(b)]
        return lst   

def stream_compression_coder(orginal_image):    

    ################################################################################################################
    # Kompresja strumieniowa RLE - koder() :
    # wejście: obraz (gs/rgb), (h,w,d) LUB obraz 2D (h,w)
    #                                 -> zwraca:
    #                                 pojedynczą zmienną (listę) -> v[] (Zawiera informacje o wymiarach org. obrazu)

    if(len(orginal_image.shape)>2):
    
        img_height = orginal_image.shape[0]
        img_width = orginal_image.shape[1]
        img_num_of_dim = orginal_image.shape[2]      

        a = ThreeD(img_num_of_dim, img_width, img_height)       

        for i in tqdm(range(img_height)):
            for j in range(img_width): 
                for w in range(img_num_of_dim):
                   
                    a[i][j][w] = orginal_image[i,j,w] 

        v = []
        
        n_col = len(a)       
        n_row = len(a[0])   
        n_dim = len(a[0][0]) 

        v.append(n_col)
        v.append(n_row)
        v.append(n_dim) 
      
        aa = [] 

        for i in tqdm(range(len(a))): 
            for j in range(len(a[0])): 
                for w in range(len(a[0][0])): 

                    aa.append(a[i][j][w])   

        counter = 1

        for i in tqdm(range(len(aa))):  

            if(i+1<len(aa)):
                if(aa[i] == aa[i+1]):
                    counter +=1
                else:               
                    v.append(counter) 
                    v.append(aa[i])   
                    counter = 1       
            else:
                v.append(counter)            
                v.append(aa[i])    

        return v 

    else:

        img_height = orginal_image.shape[0]
        img_width = orginal_image.shape[1]                 

        a = TwoD(img_width, img_height)     

        for i in range(img_height):
            for j in range(img_width): 
              
                a[i][j] = orginal_image[i,j] 

        v = []
        
        n_col = len(a)       
        n_row = len(a[0])           

        v.append(n_col)
        v.append(n_row)
      
        aa = [] 
       
        for i in range(len(a)): 
            for j in range(len(a[0])): 
               
                aa.append(a[i][j])   

        counter = 1
        
        for i in range(len(aa)):  

            if(i+1<len(aa)):
                if(aa[i] == aa[i+1]):
                    counter +=1
                else:               
                    v.append(counter) 
                    v.append(aa[i])   
                    counter = 1       
            else:
                v.append(counter)            
                v.append(aa[i])   

        return v    

def stream_compression_decoder(v): 

    ##########################################################################################################
    # Kompresja strumieniowa - dekoder() :
    # Zwraca oryginalną postać danych, które podległy kompresji strumieniowej    

    img_height = v[0]
    img_width = v[1]
       
    target = TwoD(img_width, img_height)   
   
    v = v[2:]
   
    aa =  []     

    k = 0
    x = 1 
   
    for i in range(len(v)):
       
        if(((k+2)<len(v)) and ((x+2)<len(v))):

            counter = int(v[k])
            value = v[x]

            for j in range(counter):

                aa.append(value)

            k+= 2 
            x+= 2

        else:

            counter = int(v[k])
            value = v[x]
            
            for j in range(counter):

                aa.append(value)

            break  

    k = 0
    
    for i in range(img_height):
        for j in range(img_width):            

            target[i][j] = aa[k]

            k+=1      

    return target 

def jpeg_algorithm(YCrCb, Qy, Qc, ch_sub_type, quant_table):      
    
    YCrCb_h = YCrCb.shape[0] 

    YCrCb_w = YCrCb.shape[1] 

    
    if(ch_sub_type == "4:2:2"):

        YCrCb = chroma_subsampling(YCrCb, "4:2:2") 

    else:

        YCrCb = chroma_subsampling(YCrCb, "4:4:4")

    result = np.zeros((YCrCb_h,YCrCb_w,3))

    result_block = np.zeros((8,8,3)) 

    block = np.zeros((8,8))    

    n = 0
    m = 0

    for l in range(int((YCrCb_h/8)**2)):       

        for k in range(3): 
            for w in range(1): 
                for i in range(8):
                    for j in range(8):                        

                        block[i,j] = YCrCb[i+n,j+m,k]            

            block = dct2(block) 

            block = quantization(block, Qy, Qc, k, quant_table) 

            block = zigzag(block).astype(int) 
           
            block = np.reshape(block, (8,8))
           
            block = stream_compression_coder(block)               

            block = stream_compression_decoder(block)            

            block = np.reshape(block,(8,8))
            block = np.reshape(block,(64,))            

            block = zigzag(block)

            block = dequantization(block, Qy, Qc, k, quant_table)           

            block = idct2(block)       

            for w in range(1): 
                for i in range(8):
                    for j in range(8):

                        result_block[i,j,k] = block[i,j]  

        result_block[:,:,0] = result_block[:,:,0] + 128
        result_block[:,:,1] = result_block[:,:,1] + 128
        result_block[:,:,2] = result_block[:,:,2] + 128       

        result_block = np.clip(result_block, 0, 255)

        block_RGB = cv2.cvtColor(result_block.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
       
        for i in range(8): 
            for j in range(8):
                for w in range(3):

                    result[i+n,j+m,w] = block_RGB[i,j,w].astype(int)  
                     
        for x in range(1):
            for y in range(1):

                if(m != YCrCb_w):

                    m = m + 8

                if(m == YCrCb_w):

                    m = 0

                    if(n != YCrCb_h):
                    
                        n = n + 8

                if(n == YCrCb_h):

                    n = 0        

    return result.astype(int)

Qy= np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ])

Qc= np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ])

jpeg_image = jpeg_algorithm(YCrCb, Qy, Qc, "4:2:2", True)

print("\n jpeg_image = \n", jpeg_image)
print("\n jpeg_image.shape = \n", jpeg_image.shape)
print("\n len(jpeg_image) = \n", len(jpeg_image))
print("\n jpeg_image.dtype = \n", jpeg_image.dtype)
print("\n type(jpeg_image) = \n", type(jpeg_image))

plt.imshow(block)
plt.show()