import numpy as np
import cv2
import matplotlib.pyplot as plt 
from PIL import Image 
from numpy import random
import math 
from math import sqrt
import sys

def closest_color(color_value, color_palette, w):
   
    #r, g, b = color_values
    
    if(color_palette.shape[1] < 3): # gray_scale        
      
        c = color_value
        color_diffs = []
        for color in color_palette:
            cr = color
            color_diff = sqrt(abs(c - cr)**2)
            color_diffs.append((color_diff, color))    
        return min(color_diffs)[1]
       
    else:   # rgb ...
        if(w == 0):    
            c = color_value
            color_diffs = []
            for color in color_palette[:,0]:
                cr = color
                color_diff = sqrt(abs(c - cr)**2)
                color_diffs.append((color_diff, color))    
            return min(color_diffs)[1]
        if(w == 1):    
            c = color_value
            color_diffs = []
            for color in color_palette[:,1]:
                cg = color
                color_diff = sqrt(abs(c - cg)**2)
                color_diffs.append((color_diff, color))    
            return min(color_diffs)[1]
        if(w == 2):    
            c = color_value
            color_diffs = []
            for color in color_palette[:,2]:
                cb = color
                color_diff = sqrt(abs(c - cb)**2)
                color_diffs.append((color_diff, color))    
            return min(color_diffs)[1]  

def generate_color_palete(img1, number_of_colors):
    #def generate_color_palete(img1, bit_res):

    # ta funkcja tworzy paletę barw danego obrazu, zredunkowaną do ilości kolorów określoną w parametrze number_of_colors

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]   

    if(len(img1.shape)<3):        
        
        # gray_scale ...

        color_values = np.zeros((img1_height*img1_width,1),dtype=float)

        for i in range(img1_height):
            for j in range(img1_width):           

                # tablica przechowująca wartości kolorów obrazu :        
                color_values[i*img1.shape[1]+j] = img1[i,j]

        color_values_unique = np.unique(color_values)

        # color values unique - przechowuje wartości kolorów pikseli oryginalnego obrazu (gray_scale)
        
        color_values_unique = np.reshape(color_values_unique, (len(color_values_unique), 1))

        color_values = color_values_unique                           

        color_palette = np.linspace(min(color_values), max(color_values), number_of_colors) #

        return color_palette
    else:   
        
        # rgb ...        

        img1_num_of_dim = img1.shape[2]

        R = img1[:,:,0]
        G = img1[:,:,1]
        B = img1[:,:,2]        

        color_values_R = np.zeros((img1_height*img1_width,1),dtype=float)
        color_values_G = np.zeros((img1_height*img1_width,1),dtype=float)
        color_values_B = np.zeros((img1_height*img1_width,1),dtype=float)

        for i in range(img1_height):
                for j in range(img1_width):         
                
                    color_values_R[i*img1_width+j] = R[i,j]
                    color_values_G[i*img1_width+j] = G[i,j]
                    color_values_B[i*img1_width+j] = B[i,j]          

        color_palette = np.zeros((min(len(color_values_R), len(color_values_G), len(color_values_B)),3),dtype=float)       

        for i in range(color_palette.shape[0]):
            for j in range(color_palette.shape[1]):

                if(j == 0):
                    color_palette[i,j] =  color_values_R[i]

                if(j == 1):
                    color_palette[i,j] =  color_values_G[i]

                if(j == 2):
                    color_palette[i,j] =  color_values_B[i]       

        color_palette = np.unique(color_palette, axis=0)       

        col_1 = np.linspace(min(color_palette[:,0]), max(color_palette[:,0]), number_of_colors)
        col_2 = np.linspace(min(color_palette[:,1]), max(color_palette[:,1]), number_of_colors)
        col_3 = np.linspace(min(color_palette[:,2]), max(color_palette[:,2]), number_of_colors)        

        color_palette = np.zeros((number_of_colors,3),dtype=float)        

        for i in range(number_of_colors):

            for j in range(3):

                if(j == 0):
                    color_palette[i,j] =  col_1[i]

                if(j == 1):
                    color_palette[i,j] =  col_2[i]

                if(j == 2):
                    color_palette[i,j] =  col_3[i]

        return color_palette


def quantization(img1, color_palette):

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]   

    if(len(img1.shape)<3):        

        target = np.zeros((img1_height,img1_width,1),dtype=float)
       
        for i in range(img1_height):
            for j in range(img1_width):                

                pixel_value = img1[i,j]                        
                
                closest_color_ = closest_color(pixel_value, color_palette, 1)

                target[i,j] = closest_color_

        return target

    else:     

        img1_num_of_dim = img1.shape[2]

        target = np.zeros((img1_height,img1_width,img1_num_of_dim),dtype=float)

        for w in range(img1_num_of_dim):
            for i in range(img1_height):
                for j in range(img1_width):                    

                    pixel_value = img1[i,j,w]                        
                    
                    closest_color_ = closest_color(pixel_value, color_palette, w)

                    target[i,j,w] = closest_color_

        return target

def binarization(img1, threshold, auto): # trzeci parametr - określa czy obliczyć próg automatycznie (1), czy użyc parametru podanego przez usera (0)

    # Funkcja realizująca binaryzację obrazu w skali odcieni szarości

    # obraz gray_scale -> obraz binarny (0,1)

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]

    target = np.zeros((img1_height,img1_width,1),dtype=float) 
    
    if(auto == 1): # chcemy obliczyć próg automatycznie ...

        color_values = np.zeros((img1_height*img1_width,1),dtype=float)       
    
        for i in range(img1_height):
            for j in range(img1_width):          
       
                if(len(img1.shape)<3):
                    color_values[i*img1_width+j] = img1[i,j]
                else: 
                    color_values[i*img1_width+j] = np.mean([img1[i,j,0], img1[i,j,1], img1[i,j,2]])
                    
        color_values_unique = np.unique(color_values)
        
        color_values_unique = np.reshape(color_values_unique, (len(color_values_unique), 1))

        color_values = color_values_unique

        color_values 
        
        thres_min = min(color_values)
        thres_max = max(color_values)

        threshold = (thres_min + thres_max) / 2
    
    # (próg podany przez użytkownika), binaryzacja : 
    
    for i in range(img1_height):
        for j in range(img1_width):

            if(len(img1.shape)<3):
                if(img1[i,j] > threshold):
                    target[i][j] = 1
                else:
                    target[i][j] = 0
            else:
                if(np.mean([img1[i,j,0], img1[i,j,1], img1[i,j,2]]) > threshold):
                    target[i][j] = 1
                else:
                    target[i][j] = 0

    return target

def random_dithering(img1): 

    # Funkcja realizująca dithering losowy (dla obrazu gray_scale)

    # WEJŚCIE                                        # ZWRACA :
    # obraz w skali odcieni szarości (gray_scale) ->   obraz po przepuszczeniu przez dithering losowy

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]

    random_values = np.zeros((img1_height,img1_width,1),dtype=float) # macierz losowych wartości

    target = np.zeros((img1_height,img1_width,1),dtype=float) # to będzie wyjściowy obraz (zbinaryzowany) -> tj. (0,1)
    
    # Generowanie losowych wartości obrazu : 

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            random_values[i,j] = np.random.random_sample()

    for i in range(target.shape[0]):
        for j in range(target.shape[1]):

            # sprawdzenie, czy wartość piksela w skali odcieni szarości jest większa od wygenerowanej wartości losowej 

            if(img1[i][j] > random_values[i][j]): 
                target[i][j] = 1
            else:
                target[i][j] = 0

    return target

def generate_threshold_map(matrix_number):

    # Funkcja ta zwraca macierz progowania (Threshold map) - zwaną również jako Macierz Bayera.

    # Jest ona potrzebna do zrealizowania ditheringu zorganizowanego 

    #M1 = np.array([[0, 2], [3, 1]], dtype=float)/4
    
    M1 = np.array([[0, 2], [3, 1]], dtype=float)

    # Przekształcenie mapy do przestrzeni <-0.5, 0.5> :

    M1_pre = M1.copy()

    n = int(M1.shape[0]/2) 

    print("\n n = \n", n)

    for i in range(M1_pre.shape[0]):
        for j in range(M1_pre.shape[1]): 
            
            M1_pre[i,j] = ((M1[i,j]+1) / ((2*n)**2)) - 0.5

    print("\n M1_pre = \n", M1_pre)

    # M_out - macierz którą chcemy zrówcić (M2, M4, M6 ...)
    
    M = []

    M.append(M1)

    for i in range(matrix_number):

        #M_out = np.concatenate([np.concatenate([(2*n)**2 * M1,(2*n)**2 * M1+2],axis = 1),np.concatenate([(2*n)**2 * M1+3,(2*n)**2 * M1+1],axis = 1)],axis = 0)
        M_out = np.concatenate([np.concatenate([(2*n)**2 * M[i],(2*n)**2 * M[i]+2],axis = 1),np.concatenate([(2*n)**2 * M[i]+3,(2*n)**2 * M[i]+1],axis = 1)],axis = 0)

        #M.append(M_out)

        M_out_pre = M_out.copy()

        n = int(M_out_pre.shape[0]/2)        

        for i in range(M_out_pre.shape[0]):
            for j in range(M_out_pre.shape[1]): 
                
                M_out_pre[i,j] = ((M_out[i,j]+1) / ((2*n)**2)) - 0.5

        M.append(M_out_pre)
    
    return M[matrix_number-1]

def organised_dithering(img1, M, color_palette):

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    
    if(len(img1.shape)<3):
    
        target = np.zeros((img1_height,img1_width,1),dtype=float)

        n = int(M.shape[0]/2)   

        for i in range(img1_height):
            for j in range(img1_width):        

                pixel_value = img1[i,j]
        
                color_value = closest_color(pixel_value+(1)*(M[i % 2*n, j % 2*n]+0.5), color_palette, 1)

                target[i,j] = color_value

        return target

    else:

        img1_num_of_dim = img1.shape[2]

        target = np.zeros((img1_height,img1_width,img1_num_of_dim),dtype=float)       

        n = int(M.shape[0]/2)   

        for w in range(img1_num_of_dim):
            for i in range(img1_height):
                for j in range(img1_width):                    

                    pixel_value = img1[i,j,w]
                   
                    color_value = closest_color(pixel_value+(1)*(M[i % 2*n, j % 2*n]+0.5), color_palette, w)

                    target[i,j,w] = color_value

        return target

def floyd_steinberg_dithering(img1, color_palette):

    img1_height = img1.shape[0]
    img1_width = img1.shape[1]

    if(len(img1.shape)<3):
    
        for i in range(img1_height):
            for j in range(img1_width):                

                oldpixel  = img1[i,j]

                newpixel = closest_color(oldpixel, color_palette, 1)

                img1[i,j] = newpixel

                quant_error = oldpixel - newpixel
               
                if(i<img1_height-1):
                    img1[i+1][j] = img1[i+1][j] + quant_error * 7 / 16
        
                if(i>0):
                    if(j<img1_width-1):
                        img1[i-1][j+1] = img1[i-1][j+1] + quant_error * 3 / 16  
        
                if(j<img1_width-1):
                    img1[i][j+1] = img1[i][j+1] + quant_error * 5 / 16        
            
                if((i<img1_height-1) and (j < img1_width-1)):
                    img1[i+1][j+1] = img1[i+1][j+1] + quant_error * 1 / 16
        
        return img1        

    else:        

        img1_num_of_dim = img1.shape[2]

        for w in range(img1_num_of_dim):
            for i in range(img1_height):
                for j in range(img1_width):                   

                    oldpixel  = img1[i,j,w]

                    newpixel = closest_color(oldpixel, color_palette, w)

                    img1[i,j,w] = newpixel

                    quant_error = oldpixel - newpixel

                    if(i<img1_height-1):
                        img1[i+1][j][w] = img1[i+1][j][w] + quant_error * 7 / 16
        
                    if(i>0):
                        if(j<img1_width-1):
                            img1[i-1][j+1][w] = img1[i-1][j+1][w] + quant_error * 3 / 16  
        
                    if(j<img1_width-1):
                        img1[i][j+1][w] = img1[i][j+1][w] + quant_error * 5 / 16
        
            
                    if((i<img1_height-1) and (j < img1_width-1)):
                        img1[i+1][j+1][w] = img1[i+1][j+1][w] + quant_error * 1 / 16   

        return img1  
        
##########################################################################################################

# KWANTYZACJA RGB :

img1 = plt.imread('land3.png')

img1_height = img1.shape[0]
img1_width = img1.shape[1]

# bit_res = 3 # rozdzielczość bitowa -> ilość bitów na jakich zapisany jest każdy piksel 

# Na podstawie tej zmiennej tworzona jest paleta barw : 

number_of_colors = 4   # określa ilość kolorów, która zostanie wygenerowana w palecie barw

print("\n img1 = \n", img1)
print("\n img1.dtype = \n", img1.dtype)
print("\n img1.shape = \n", img1.shape)

color_palette = generate_color_palete(img1, number_of_colors)

print("\n color_palette (rgb) = \n\n", color_palette)
print("\n color_palett.dtype = \n", color_palette.dtype)
print("\n color_palette.shape = ", color_palette.shape)

print("\n Kwantyzacja (rgb) -> \n")

target = quantization(img1, color_palette)

plt.subplot(1,2,1)
plt.imshow(img1)
plt.title('img1')


plt.subplot(1,2,2)
plt.imshow(target)
plt.title('target')

plt.show()

##########################################################################################

# KWANTYZACJA GS :

R = img1[:,:,0]
G = img1[:,:,1]
B = img1[:,:,2]

Y = 0.299 * R + 0.587 * G + 0.114 * B

#img1 = Y.copy()
                  
color_palette_gs = generate_color_palete(Y, number_of_colors) 

print("\n color_palette_gs = \n\n", color_palette_gs)
print("\n color_palette_gs.dtype = \n", color_palette_gs.dtype)
print("\n color_palette_gs.shape = \n", color_palette_gs.shape)

print("\n Kwantyzacja (gray_scale) -> \n")

target = quantization(Y, color_palette_gs)

print("target = \n", target)

plt.subplot(1,2,1)
plt.imshow(Y, cmap='gray')
plt.title('img1')

plt.subplot(1,2,2)
plt.imshow(target, cmap='gray')
plt.title('target')

plt.show()

##########################################################################################

# BINARYZACJA ( ✓ RGB,  ✓ GS) :

print("\n BINARYZACJA (rgb / gray_scale) -> \n")

print("\n Y = \n\n", Y)
print("\n Y.dtype = \n", Y.dtype)
print("\n Y.shape = \n", Y.shape)

thres = 0.5 # próg binaryzacji -> ustawiony ręcznie, jednak funkcja wykonująca binaryzację może także obliczyć ten próg automatycznie (jeśli trzeci parametr ustawiony będzie na "1")

binary_image = binarization(Y, thres, 1)

plt.subplot(1,2,1)
plt.imshow(Y, cmap='gray')
#plt.imshow(img1)
#plt.title('Y gray_scale')
plt.title('orginal')

plt.subplot(1,2,2)
plt.imshow(binary_image, cmap='gray')
plt.title('binary_image')

plt.show()

##########################################################################################

# DITHERING LOSOWY ( ✓ GS) :

print("\n DITHERING LOSOWY (GS) -> \n")

binary_image_random_dithering = random_dithering(Y) # przyjmuje jako parametr tylko obraz w skali odcieni szarości

plt.subplot(1,3,1)
plt.imshow(Y, cmap='gray')
plt.title('orginal image')

plt.subplot(1,3,2)
plt.imshow(binary_image, cmap='gray')
plt.title('binary_image')

plt.subplot(1,3,3)
plt.imshow(binary_image_random_dithering, cmap='gray')
plt.title('binary_image_random_dithering')
plt.show()

##########################################################################################

# DITHERING ZORGANIZOWANY - M2 (4x4) ( ✓ GS) :

print("\n  DITHERING ZORGANIZOWANY (GS) -> \n")

#M2 = generate_threshold_map(3) # 2 -> M2 (4x4), 3 -> M4 (8x8), ... itd
M2 = generate_threshold_map(2) # 2 -> M2 (4x4), 3 -> M4 (8x8), ... itd

print("\n  M2 -> \n", M2)
print("\n  M2 -> \n", M2.shape)

#target = np.zeros((img1_height,img1_width,1),dtype=float)

target = organised_dithering(Y, M2, color_palette_gs)

plt.subplot(1,2,1)
plt.imshow(Y, cmap='gray')
plt.title('img1')


plt.subplot(1,2,2)
plt.imshow(target, cmap='gray')
plt.title('target')

plt.show()

#exit()

print("\n##############################################################")

print("\n  DITHERING ZORGANIZOWANY (RGB) -> \n")

#print("\n img1 = \n", img1)
#print("\n img1.dtype = \n", img1.dtype)
#print("\n img1.shape = \n", img1.shape)

color_palette = generate_color_palete(img1, number_of_colors)

#print("\n color_palette (rgb) = \n\n", color_palette)
#print("\n color_palett.dtype = \n", color_palette.dtype)
#print("\n color_palette.shape = ", color_palette.shape)

target = organised_dithering(img1, M2, color_palette)

plt.subplot(1,2,1)
plt.imshow(img1)
plt.title('img1')


plt.subplot(1,2,2)
plt.imshow(target)
plt.title('target')

plt.show()

##########################################################################################

# DITHERING FLOYD–STEINBERGA ( ✓ GS, ✓ RGB) :

print("\n  DITHERING FLOYD–STEINBERGA (GS) -> \n")

img1_org = img1.copy()

img_gs = floyd_steinberg_dithering(Y, color_palette_gs)

img_rgb = floyd_steinberg_dithering(img1, color_palette)

plt.subplot(1,3,1)
plt.imshow(img1_org)
plt.title('img1')

plt.subplot(1,3,2)
plt.imshow(img_gs, cmap='gray')
plt.title('Floyd-Steinberg (gs)')

plt.subplot(1,3,3)
plt.imshow(img_rgb)
plt.title('Floyd-Steinberg (rgb)')

plt.show()
exit()

##########################################################################################

img1 = plt.imread('test21.png')
img1_height = img1.shape[0]
img1_width = img1.shape[1]

# Na podstawie tej zmiennej tworzona jest paleta barw : 

#number_of_colors = 4   # określa ilość kolorów, która zostanie wygenerowana w palecie barw

print("\n img1 = \n", img1)
print("\n img1.dtype = \n", img1.dtype)
print("\n img1.shape = \n", img1.shape)

#color_palette1 = generate_color_palete(img1, 2) #1 bit
#color_palette2 = generate_color_palete(img1, 4) # 2
#color_palette3 = generate_color_palete(img1, 8) # 3
#color_palette4 = generate_color_palete(img1, 16) # 4
#color_palette5 = generate_color_palete(img1, 32) # 5
#color_palette6 = generate_color_palete(img1, 64) # 6
#color_palette7 = generate_color_palete(img1, 128)# 7
#color_palette8 = generate_color_palete(img1, 256) # 8

#color_palette = generate_color_palete(img1, number_of_colors) # paleta rgb

#print("\n color_palette (rgb) = \n\n", color_palette)
#print("\n color_palett.dtype = \n", color_palette.dtype)
#print("\n color_palette.shape = ", color_palette.shape)

#R = img1[:,:,0]
#G = img1[:,:,1]
#B = img1[:,:,2]

#Y = 0.299 * R + 0.587 * G + 0.114 * B

#img1 = Y.copy()
img1_org = img1.copy()
                  
#color_palette_gs = generate_color_palete(Y, number_of_colors) # paleta gs
#color_palette_gs = generate_color_palete(img1, number_of_colors) # paleta gs

#color_palette_gs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

#color_palette_gs = np.array([[0.396, 0.454, 0.196], 
#                          [0.298, 0.505, 0.764], 
#                          [0.239, 0.349, 0.670], 
#                          [0.176, 0.219, 0.227], 
#                          [0.764, 0.803, 0.901]])

color_palette_gs = np.array([[0.274, 0.466, 0.784], 
                          [0.321, 0.396, 0.145], 
                          [0.184, 0.317, 0.619], 
                          [0.435, 0.462, 0.196], 
                          [0.164, 0.2, 0.207],
                          [0.341, 0.517, 0.756],
                          [0.6, 0.682, 0.815],
                          [0.901, 0.909, 0.980],
                          [0.254, 0.282, 0.2],
                          [0.250, 0.317, 0.411]                             
                          ])

#print("\n color_palette_gs = \n\n", color_palette_gs)
#print("\n color_palette_gs.dtype = \n", color_palette_gs.dtype)
#print("\n color_palette_gs.shape = \n", color_palette_gs.shape)

target_q1 = quantization(img1, color_palette_gs) # Kwantyzacja 

thres = 0.5 
#target_q2 = binarization(img1, thres, 1)   # Binaryzacja

#target_q3 = random_dithering(img1)  # Dithering losowy

M2 = generate_threshold_map(2) # 2 -> M2 (4x4), 3 -> M4 (8x8), ... itd

target_q4 = organised_dithering(img1, M2, color_palette_gs) # Dithering zorganizowany
target_q5 = floyd_steinberg_dithering(img1, color_palette_gs) # Dithering FLoyda-Steinberga

#target_q6 = quantization(img1, color_palette6) # Kwantyzacja 
#target_q7 = quantization(img1, color_palette7) # Kwantyzacja 
#target_q8 = quantization(img1, color_palette8) # Kwantyzacja 

#thres = 0.5 
#binary_image = binarization(Y, thres, 1)   # Binaryzacja

#binary_image_random_dithering = random_dithering(Y) # Dithering losowy


#M2 = generate_threshold_map(2) # 2 -> M2 (4x4), 3 -> M4 (8x8), ... itd
#print("\n  M2 -> \n", M2)
#print("\n  M2 -> \n", M2.shape)

#target_dit_org = organised_dithering(Y, M2, color_palette_gs) # Dithering zorganizowany


#img1_org = img1.copy()

#target_dit_fs = floyd_steinberg_dithering(Y, color_palette_gs) # Dithering FLoyda-Steinberga
    #img_rgb = floyd_steinberg_dithering(img1, color_palette)

###########################
plt.subplot(1,4,1)
plt.imshow(img1_org)
plt.title('Orginal')

plt.subplot(1,4,2)
plt.imshow(target_q1)
plt.title('Kwantyzacja - własna paleta 10 kolorów')

#plt.subplot(1,5,3)
#plt.imshow(target_q3, cmap='gray')
#plt.title('2 bity - metoda losowa')

plt.subplot(1,4,3)
plt.imshow(target_q4)
plt.title('Dithering zorganizowany')

plt.subplot(1,4,4)
plt.imshow(target_q5)
plt.title('Dithering Floyd-Steinberga')

plt.show()
exit()

plt.subplot(1,4,1)
plt.imshow(target_q4, cmap='gray')
plt.title('4 bity')

plt.subplot(1,4,2)
plt.imshow(target_q3, cmap='gray')
plt.title('3 bity')

plt.subplot(1,4,3)
plt.imshow(target_q2, cmap='gray')
plt.title('2 bity')

plt.subplot(1,4,4)
plt.imshow(target_q1, cmap='gray')
plt.title('1 bit')

plt.show()
exit()