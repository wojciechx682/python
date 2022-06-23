import numpy as np
import cv2
import matplotlib.pyplot as plt 
from PIL import Image 
from numpy import random

import math 
import sys

def nearest_neighbor_interpolation(img1, sf):
    
    img1_height = img1.shape[0] 
    img1_width = img1.shape[1] 

    newHeight = img1_height * sf    
    newWidth = img1_width * sf    

    newHeight = int(newHeight)
    newWidth = int(newWidth)
    num_of_dim = img1.shape[2]

    target = np.zeros((newHeight,newWidth,num_of_dim),dtype=float)

    target_height = target.shape[0] 
    target_width = target.shape[1]  
    target_dim = target.shape[2]    

    for w in range(target_dim): 
        for i in range(target_height):
            for j in range(target_width):

                coordYtarget = i;
                coordXtarget = j;
                
                coordYsource = round(coordYtarget / sf)
                coordXsource = round(coordXtarget / sf)           

                if (coordXsource >= img1.shape[1]):
                    coordXsource = img1.shape[1]-1 

                if (coordYsource >= img1.shape[0]):
                    coordYsource = img1.shape[0]-1        

                target[i,j,w] = img1[coordYsource,coordXsource,w]

    return target

def bilinear_interpolation(img1, sf):       

    img1_height = img1.shape[0] 
    img1_width = img1.shape[1] 

    newHeight = img1_height * sf    
    newWidth = img1_width * sf    

    newHeight = int(newHeight)
    newWidth = int(newWidth)
    num_of_dim = img1.shape[2]

    target = np.zeros((newHeight,newWidth,num_of_dim),dtype=float)

    target_height = target.shape[0] 
    target_width = target.shape[1] 
    target_dim = target.shape[2]   

    for w in range(target_dim): 
        for i in range(target_height):
            for j in range(target_width):

                coordYtarget = i 
                coordXtarget = j               

                Q11_Y = math.floor(coordYtarget/sf)
        
                Q11_X = math.floor(coordXtarget/sf)

                Q12_Y = math.floor(coordYtarget/sf)

                if(coordXtarget == 0):                   

                    Q12_X = 1 
                else:
                    Q12_X = math.ceil(coordXtarget/sf)

                    if(Q12_X >= img1.shape[1]):
                        Q12_X = math.floor(coordXtarget/sf)

                if(coordYtarget == 0):      

                    Q21_Y = 1
                else:
                    Q21_Y = math.ceil(coordYtarget/sf)

                    if(Q21_Y >= img1.shape[0]):
                        Q21_Y = math.floor(coordYtarget/sf)                

                Q21_X  = math.floor(coordXtarget/sf)

                if(coordYtarget == 0):      

                    Q22_Y = 1 
                else:
                    Q22_Y = math.ceil(coordYtarget/sf)
                        
                    if(Q22_Y >= img1.shape[0]):
                        Q22_Y = math.floor(coordYtarget/sf)

                if(coordXtarget == 0):      

                    Q22_X = 1 
                else:
                    Q22_X = math.ceil(coordXtarget/sf)

                    if(Q22_X >= img1.shape[1]):
                        Q22_X = math.floor(coordXtarget/sf)

                #x_ = coordXtarget / sf
                #y_ = coordYtarget / sf

                x_ = 1/sf    
                y_ = 1-(1/sf)
                    
                target[i,j,w] = (img1[Q11_Y,Q11_X,w]*(1-x_)*(1-y_) + img1[Q21_Y,Q21_X,w]*x_*(1-y_) + img1[Q12_Y,Q12_X,w]*(1-x_)*y_ + img1[Q22_Y,Q22_X,w]*x_*y_)

    return target

def average_interpolation(img1, sf):    
    
    img1_height = img1.shape[0] 
    img1_width = img1.shape[1] 

    newHeight = img1_height * sf    
    newWidth = img1_width * sf    

    newHeight = int(newHeight)
    newWidth = int(newWidth)
    num_of_dim = img1.shape[2]

    target = np.zeros((newHeight,newWidth,num_of_dim),dtype=float)

    target_height = target.shape[0] 
    target_width = target.shape[1]  
    target_dim = target.shape[2]   

    for w in range(target_dim): 
        for i in range(target_height):
            for j in range(target_width):                

                Y_r_bottom = round(i * (img1_height/int(img1_height*sf))) 
                Y_r_top =  round(i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf))              
                X_r_bottom =  round(j * (img1_width/int(img1_width*sf))) 
                X_r_top =  round(j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf))
                             
                if(Y_r_bottom <= 0):
                    Y_r_bottom = 0
                if(Y_r_top >= img1_height):
                    Y_r_top = img1_height-1

                if(X_r_bottom <= 0):
                    X_r_bottom = 0
                if(X_r_top >= img1_width):
                    X_r_top = img1_width-1

                Y_r_bottom1 = np.floor(i * (img1_height/int(img1_height*sf)))
                Y_r_top1 =  np.ceil(i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf))

                X_r_bottom1 =  np.floor(j * (img1_width/int(img1_width*sf)))
                X_r_top1 =  np.ceil(j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf))

                Y_r_bottom2 = i * (img1_height/int(img1_height*sf))
                Y_r_top2 =  i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf)
               
                X_r_bottom2 =  j * (img1_width/int(img1_width*sf))
                X_r_top2 =  j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf)                

                if(X_r_bottom1 == X_r_top1):
                    X_range = np.array(np.arange(X_r_bottom1, X_r_top1+1, 1, dtype=int))

                Y_range = np.array(np.arange(Y_r_bottom1, Y_r_top1, 1, dtype=int))

                X_range = np.array(np.arange(X_r_bottom1, X_r_top1, 1, dtype=int))

                if(X_r_bottom1 == X_r_top1):
                    X_range = np.array(np.arange(X_r_bottom1, X_r_top1+1, 1, dtype=int))

                if(Y_r_bottom1 == Y_r_top1):
                    Y_range = np.array(np.arange(Y_r_bottom1, Y_r_top1+1, 1, dtype=int))

                P = []            

                for k in range(Y_range.size):
                    for l in range(X_range.size):                        
                        P.append([Y_range[k],X_range[l]])  
                
                otoczenie = []               

                for k in range(len(P)):                 

                    if(P[k][0] >= img1_height):
                        if(P[k][1] >= img1_width):
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1]-1,w])
                        else:
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1],w])
                    elif(P[k][1] >= img1_width):                       
                        
                        if(P[k][0] >= img1_height):
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1]-1,w])
                        else:
                            otoczenie.append(img1[P[k][0],P[k][1]-1,w])
                    else:
                        otoczenie.append(img1[P[k][0],P[k][1],w])  

                target[i,j,w] = np.mean(otoczenie)

    return target

def median_interpolation(img1, sf):   

    img1_height = img1.shape[0] 
    img1_width = img1.shape[1] 

    newHeight = img1_height * sf    
    newWidth = img1_width * sf

    newHeight = int(newHeight)
    newWidth = int(newWidth)
    num_of_dim = img1.shape[2]

    target = np.zeros((newHeight,newWidth,num_of_dim),dtype=float)

    target_height = target.shape[0] 
    target_width = target.shape[1]  
    target_dim = target.shape[2]    

    for w in range(target_dim): 
        for i in range(target_height):
            for j in range(target_width):                

                Y_r_bottom = round(i * (img1_height/int(img1_height*sf))) 
                Y_r_top =  round(i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf))    

                X_r_bottom =  round(j * (img1_width/int(img1_width*sf))) 
                X_r_top =  round(j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf))        

                if(Y_r_bottom <= 0):
                    Y_r_bottom = 0
                if(Y_r_top >= img1_height):
                    Y_r_top = img1_height-1

                if(X_r_bottom <= 0):
                    X_r_bottom = 0
                if(X_r_top >= img1_width):
                    X_r_top = img1_width-1

                Y_r_bottom1 = np.floor(i * (img1_height/int(img1_height*sf)))
                Y_r_top1 =  np.ceil(i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf))

                X_r_bottom1 =  np.floor(j * (img1_width/int(img1_width*sf)))
                X_r_top1 =  np.ceil(j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf))

                Y_r_bottom2 = i * (img1_height/int(img1_height*sf))
                Y_r_top2 =  i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf)
               
                X_r_bottom2 =  j * (img1_width/int(img1_width*sf))
                X_r_top2 =  j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf)               

                if(X_r_bottom1 == X_r_top1):
                    X_range = np.array(np.arange(X_r_bottom1, X_r_top1+1, 1, dtype=int))

                Y_range = np.array(np.arange(Y_r_bottom1, Y_r_top1, 1, dtype=int))

                X_range = np.array(np.arange(X_r_bottom1, X_r_top1, 1, dtype=int))

                if(X_r_bottom1 == X_r_top1):
                    X_range = np.array(np.arange(X_r_bottom1, X_r_top1+1, 1, dtype=int))

                if(Y_r_bottom1 == Y_r_top1):
                    Y_range = np.array(np.arange(Y_r_bottom1, Y_r_top1+1, 1, dtype=int))
               
                P = []                  

                for k in range(Y_range.size):
                    for l in range(X_range.size):                        
                        P.append([Y_range[k],X_range[l]])  
                        
                otoczenie = []              

                for k in range(len(P)):                 

                    if(P[k][0] >= img1_height):
                        if(P[k][1] >= img1_width):
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1]-1,w])
                        else:
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1],w])
                    elif(P[k][1] >= img1_width):
                    
                        if(P[k][0] >= img1_height):
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1]-1,w])
                        else:
                            otoczenie.append(img1[P[k][0],P[k][1]-1,w])
                    else:
                        otoczenie.append(img1[P[k][0],P[k][1],w]) 
              
                target[i,j,w] = np.median(otoczenie)

    return target

def weighted_average(img1, sf):   

    img1_height = img1.shape[0] 
    img1_width = img1.shape[1] 

    newHeight = img1_height * sf    
    newWidth = img1_width * sf

    newHeight = int(newHeight)
    newWidth = int(newWidth)
    num_of_dim = img1.shape[2]

    target = np.zeros((newHeight,newWidth,num_of_dim),dtype=float)

    target_height = target.shape[0] 
    target_width = target.shape[1] 
    target_dim = target.shape[2]   

    a = []    

    v = img1_width / target_width
    j_ = round((img1_width)+(round(img1_width*sf)))

    k = 0
    y = 1 
    
    for i in range(j_):    

        if(k<(y*v)):
            a.append(k)
            k+=1
        else:        
            a.append(y*v)
            y+=1
    
    a = list(dict.fromkeys(a)) 

    value = img1_width

    counter = 0

    while(value > 0):        
        counter += 1
        value = value - v
    if(value>0): 
        value = 0

    counter -= 1    

    aa = []   

    ile_czesci = int(img1_width / v)

    for i in range(counter + 1):        
        aa.append(a[i*int((len(a)/ile_czesci)):(i+1)*int((len(a)/ile_czesci))+1])    

    for w in range(target_dim):
        for i in range(target_height):
            for j in range(target_width):

                Y_r_bottom = round(i * (img1_height/int(img1_height*sf))) 
                Y_r_top =  round(i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf))      
                X_r_bottom =  round(j * (img1_width/int(img1_width*sf))) 
                X_r_top =  round(j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf))
            
                if(Y_r_bottom <= 0):
                    Y_r_bottom = 0
                if(Y_r_top >= img1_height):
                    Y_r_top = img1_height-1

                if(X_r_bottom <= 0):
                    X_r_bottom = 0
                if(X_r_top >= img1_width):
                    X_r_top = img1_width-1               

                Y_r_bottom1 = np.floor(i * (img1_height/int(img1_height*sf)))
                Y_r_top1 =  np.ceil(i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf))               

                X_r_bottom1 =  np.floor(j * (img1_width/int(img1_width*sf)))
                X_r_top1 =  np.ceil(j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf))              

                Y_r_bottom2 = i * (img1_height/int(img1_height*sf))
                Y_r_top2 =  i * (img1_height/int(img1_height*sf)) + img1_height/int(img1_height*sf)               

                X_r_bottom2 =  j * (img1_width/int(img1_width*sf))
                X_r_top2 =  j * (img1_width/int(img1_width*sf)) + img1_width/int(img1_width*sf)
                        
                if(X_r_bottom1 == X_r_top1):
                    X_range = np.array(np.arange(X_r_bottom1, X_r_top1+1, 1, dtype=int))

                Y_range = np.array(np.arange(Y_r_bottom1, Y_r_top1, 1, dtype=int))

                X_range = np.array(np.arange(X_r_bottom1, X_r_top1, 1, dtype=int))

                if(X_r_bottom1 == X_r_top1):
                    X_range = np.array(np.arange(X_r_bottom1, X_r_top1+1, 1, dtype=int))

                if(Y_r_bottom1 == Y_r_top1):
                    Y_range = np.array(np.arange(Y_r_bottom1, Y_r_top1+1, 1, dtype=int))              

                P = [] 
        
                for k in range(Y_range.size):
                    for l in range(X_range.size):                       
                         P.append([Y_range[k],X_range[l]])              

                otoczenie = []

                for k in range(len(P)):     

                    if(P[k][0] >= img1_height):
                        if(P[k][1] >= img1_width):
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1]-1,w])
                        else:
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1],w])
                    elif(P[k][1] >= img1_width):                       
                        
                        if(P[k][0] >= img1_height):
                            otoczenie.append(img1[P[k][0]-1 ,P[k][1]-1,w])
                        else:
                            otoczenie.append(img1[P[k][0],P[k][1]-1,w])
                    else:
                        otoczenie.append(img1[P[k][0],P[k][1],w]) 
    
                O_W = 0     
                suma_wag = 0
                
                for k in range(len(P)):                

                    a_Y = aa[i] 
                    a_X = aa[j]              

                    w_Y_old = (P[k][0])
                    w_X_old = (P[k][1])      
               
                    if(w_Y_old >= (len(a_Y)-1)):                        
                        w1 = a_Y[len(a_Y)-1] - a_Y[len(a_Y)-2]
                    else:
                        w1 = a_Y[w_Y_old+1] - a_Y[w_Y_old]              

                    if(w_X_old >= (len(a_X)-1)):
                       
                        w2 = a_X[len(a_X)-1] - a_X[len(a_X)-2]
                    else:
                        w2 = a_X[w_X_old+1] - a_X[w_X_old]                             

                    O_W += (otoczenie[k] * (w1 * w2))
                    suma_wag += (w1 * w2)                  
             
                result = O_W / suma_wag
               
                target[i,j,w] = result

    return target



img1 = plt.imread('test16.png')

print(img1.shape)

sf = 0.5

target_1 = nearest_neighbor_interpolation(img1, sf)
target_2 = bilinear_interpolation(img1, sf)
target_3 = average_interpolation(img1, sf)
target_4 = median_interpolation(img1, sf)
target_5 = weighted_average(img1, sf)

y1 = 5
y2 = 80

x1 = 183
x2 = 240

#fragment_0 = img1[y1:y2,x1:x2].copy()
#fragment_1 = target_1[round(y1*sf):round(y2*sf),round(x1*sf):round(x2*sf)].copy()
#fragment_2 = target_2[round(y1*sf):round(y2*sf),round(x1*sf):round(x2*sf)].copy()
#fragment_3 = target_3[round(y1*sf):round(y2*sf),round(x1*sf):round(x2*sf)].copy()
#fragment_4 = target_4[round(y1*sf):round(y2*sf),round(x1*sf):round(x2*sf)].copy()
#fragment_5 = target_5[round(y1*sf):round(y2*sf),round(x1*sf):round(x2*sf)].copy()

plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.title('original')

plt.subplot(1, 3, 2)
plt.imshow(target_1)
plt.title('Nearest neighbor')

plt.subplot(1, 3, 3)
plt.imshow(target_2)
plt.title('Bilinear ')

plt.show()

plt.subplot(1, 3, 1)
plt.imshow(target_3)
plt.title('Average ')

plt.subplot(1, 3, 2)
plt.imshow(target_4)
plt.title('Median')

plt.subplot(1, 3, 3)
plt.imshow(target_5)
plt.title('Weighted average')

plt.show()
exit()

plt.subplot(1, 3, 1)
plt.imshow(fragment_0)
plt.title('original')

plt.subplot(1, 3, 2)
plt.imshow(fragment_1)
plt.title('Nearest neighbor')

plt.subplot(1, 3, 3)
plt.imshow(fragment_2)
plt.title('Bilinear ')

plt.show()

plt.subplot(1, 3, 1)
plt.imshow(fragment_3)
plt.title('Average ')

plt.subplot(1, 3, 2)
plt.imshow(fragment_4)
plt.title('Median')

plt.subplot(1, 3, 3)
plt.imshow(fragment_5)
plt.title('Weighted average')

plt.show()

exit()