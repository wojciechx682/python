import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack
from scipy.interpolate import interp1d
import sys
import math 
from math import sqrt
from tqdm import tqdm
import pprint
import random
from random import seed
from random import randint
import locale

from itertools import chain

def ThreeD(a, b, c):

    ##########################################################################################################
    # Funkcja tworząca listę trójwymiarową (i,j,w) - (3D):

        lst = [[['0' for col in range(a)] for col in range(b)] for row in range(c)]
        return lst   

def TwoD(a, b):

    ##########################################################################################################
    # Funkcja tworząca listę dwuwymiarową (i,j) - (2D):
    
        lst = [['0' for col in range(a)] for col in range(b)]
        return lst   

def stream_compression_coder(img):        

    ##########################################################################################################
    # Kompresja strumieniowa - koder() :
    # wejście: obraz (gs/rgb), (h,w,d)
    #                                 -> zwraca:
    #                                 pojedynczą zmienną (listę) -> v[] (Zawiera informacje o wymiarach org. obrazu)
    # Struktura danych : lista [],    # -> [ [1,2], [3,4], 5, 6, [[7,8], [9,10]] ]

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

def stream_compression_decoder(v): 

    ##########################################################################################################
    # Kompresja strumieniowa - dekoder() :
    # Zwraca oryginalną postać danych, które podległy kompresji strumieniowej

    img_height = v[0]
    img_width = v[1]
    img_n_dim = v[2]
    
    target = ThreeD(img_n_dim, img_width, img_height)   

    v = v[3:]
   
    aa =  []     

    k = 0
    x = 1 
   
    for i in tqdm(range(len(v))):
       
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
    
    for i in tqdm(range(img_height)):
        for j in range(img_width):
            for w in range(img_n_dim):        

                target[i][j][w] = aa[k]

                k+=1      

    return target  

def split_4(a):

    # wejście : a[] (lista)           

    """ 
        a =
        [[1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]]  
    """

    # wyjście : (oczekiwane, tak - przykładowo - powinny wyglądać macierze) :

    # M1 = [[1,1],[1,1]]
    # M1 = [[0,1],[1,0]]
    # M1 = [[1,0],[0,1]]
    # M1 = [[1,1],[0,1]]   

    a_is_only_one_column = False
    a_is_only_one_row = False

    if((len(a)>1) and (len(a[0]) == 1)): # to jest ok, ALE MUSISZ MIEĆ DANE W ODPOWIEDNIEJ POSTACI ! 

        a_is_only_one_column = True
    
    if((len(a)==1) and (len(a[0]) > 1)): 
    
        a_is_only_one_row = True

    if(a_is_only_one_row):
       
        new_h = 1    
        new_w = int(np.ceil(len(a[0])/2))         

        idx_w_M1 = np.arange(0,new_w,1)  
        idx_w_M2 = np.arange(new_w,len(a[0]),1) 
        idx_h_M1 = np.arange(0,new_h,1)      

        M1 = []
        M2 = []

        for i in range(2):        

            for k in range(new_h):
                for l in range(new_w): 

                    if (l > new_w): 
                        break

                    if(i==0):
                        
                        M1.append(a[0][l])

                    if(i==1):                           
                        
                        if(l+new_w<len(a[0])):
                            
                            M2.append(a[0][l+new_w])                  

        return [[M1],[M2]]
       
    if(a_is_only_one_column):

        # dane 'a' - to jeden wiersz, zwróc uwagę że 
        # idx_w_M2  - będzie wtedy równe 0 
        # (idx_w_M2 == 0)       

        # a = [[1],[0],[0],[1]] <- tak wygląda postać listy, która jest JEDNĄ KOLUMNĄ w macierzy ! 

        new_h = int(np.ceil(len(a)/2))    
        new_w = 1

        idx_w_M1 = np.arange(0,new_w,1) 
        idx_h_M1 = np.arange(0,new_h,1) # [0 1 ]        
        idx_h_M2 = np.arange(new_h,len(a),1) # [2 3]

        M1 = []
        M2 = []

        for i in range(2): 

            for k in range(new_w): 
                for l in range(new_h): 

                    if (l > new_w): 
                        break

                    if(i==0):
                            
                        M1.append([a[l][0]])

                    if(i==1):                           
                        
                        if(l+new_h<len(a)):

                            M2.append([a[l+new_h][0]])

        return [M1,M2]   

    new_h = int(np.ceil(len(a)/2)) 
    new_w = int(np.ceil(len(a[0])/2))

    idx_w_M1 = np.arange(0,new_w,1) 

    idx_w_M2 = np.arange(new_w,len(a[0]),1) 

    idx_h_M1 = np.arange(0,new_h,1) 
       
    idx_h_M2 = np.arange(new_h,len(a),1)   

    M1 = []
    M2 = []
    M3 = []
    M4 = []

    values = [] 
   
    for i in range(4): 

            for k in range(new_h):  
                for l in range(new_w): 
                    
                    if (l > new_w):
                        break

                    if(i==0):
                            
                        M1.append(a[k][l])

                    if(i==1):
                        
                        if(l+new_w<len(a[0])):

                            M2.append(a[k][l+new_w])

                    if(i==2):
                          
                        if(k+new_h<len(a)):

                            M3.append(a[k+new_h][l])

                    if(i==3):
                           
                        if((l+new_w<len(a[0]))and(k+new_h<len(a))):

                            M4.append(a[k+new_h][l+new_w])
            
    if((len(idx_h_M1) > 1) and (len(idx_w_M1) > 1)):            
        
        a_M1 = TwoD(len(idx_w_M1), len(idx_h_M1))
        
    else:
          
        a_M1 = [M1]        

    if((len(idx_h_M1) > 1) and (len(idx_w_M2) > 1)):    
        
        a_M2 = TwoD(len(idx_w_M2), len(idx_h_M1))
       
    else:
            
        a_M2 = [M2]       

    if((len(idx_h_M2) > 1) and (len(idx_w_M1) > 1)):
        
        a_M3 = TwoD(len(idx_w_M1), len(idx_h_M2))
        
    else:
           
        a_M3 = [M3]

    if((len(idx_h_M2) > 1) and (len(idx_w_M2) > 1)):
       
        a_M4 = TwoD(len(idx_w_M2), len(idx_h_M2))
        
    else:
        a_M4 = [M4]

    if((len(idx_h_M1)>1) and (len(idx_w_M1)>1) and (len(idx_h_M2)>1) and (len(idx_w_M2)>1)):

        k=0
        for i in range(len(idx_h_M1)):
            for j in range(len(idx_w_M1)):     
               
                a_M1[i][j] = M1[k] 
                k+=1
        k=0
        for i in range(len(idx_h_M1)):
            for j in range(len(idx_w_M2)):     
                   
                a_M2[i][j] = M2[k]                
                k+=1
        k=0
        for i in range(len(idx_h_M2)):
            for j in range(len(idx_w_M1)):     
               
                a_M3[i][j] = M3[k] 
                k+=1
        k=0
        for i in range(len(idx_h_M2)):
            for j in range(len(idx_w_M2)):     
                
                a_M4[i][j] = M4[k] 
                k+=1
    else:

        for i in range(4):

            if(i==0): 

                if((len(idx_h_M1) == 1) and (len(idx_w_M1) == 1)): 

                    a_M1 = [M1]

                if((len(idx_w_M1) == 1) and (len(idx_h_M1) > 1)):   

                    for x in range(len(idx_h_M1)):
                      
                        a_M1[0][x] = M1[x]

                if((len(idx_h_M1) == 1) and (len(idx_w_M1) > 1)): 

                    for x in range(len(idx_w_M1)):
                       
                        a_M1[0][x] = M1[x]

                if((len(idx_h_M1) > 1) and (len(idx_w_M1) > 1)):  

                    k = 0 
                    for x in range(len(idx_h_M1)):
                        for y in range(len(idx_w_M1)):
                            
                            a_M1[x][y] = M1[k]
                            k+=1

            if(i==1):

                if((len(idx_h_M1) == 1) and (len(idx_w_M2) == 1)):  

                    a_M2 = [M2]

                  
                if((len(idx_w_M2) == 1) and (len(idx_h_M1) > 1)):  

                    

                    for x in range(len(idx_h_M1)):
                       
                        a_M2[0][x] = M2[x]

                if((len(idx_h_M1) == 1) and (len(idx_w_M2) > 1)):  

                    for x in range(len(idx_w_M2)):
                       
                        a_M2[0][x] = M2[x] 

                if((len(idx_h_M1) > 1) and (len(idx_w_M2) > 1)):  

                    k = 0 
                    for x in range(len(idx_h_M1)):
                        for y in range(len(idx_w_M2)):
                           
                            a_M2[x][y] = M2[k]
                            k+=1

            if(i==2):

                if((len(idx_h_M2) == 1) and (len(idx_w_M1) == 1)):
                   
                    a_M3 = [M3]
                     
                if((len(idx_w_M1) == 1) and (len(idx_h_M2) > 1)): 

                    for x in range(len(idx_h_M2)):
                      
                        a_M3[0][x] = M3[x]

                if((len(idx_h_M2) == 1) and (len(idx_w_M1) > 1)):

                    for x in range(len(idx_w_M1)):
                      
                        a_M3[0][x] = M3[x]

                if((len(idx_h_M2) > 1) and (len(idx_w_M1) > 1)): 

                    k = 0 
                    for x in range(len(idx_h_M2)):
                        for y in range(len(idx_w_M1)):
                            
                            a_M3[x][y] = M3[k]
                            k+=1

            if(i==3): 

                if((len(idx_h_M2) == 1) and (len(idx_w_M2) == 1)):  

                    a_M4 = [M4]
                    
                if((len(idx_w_M2) == 1) and (len(idx_h_M2) > 1)):  

                    for x in range(len(idx_h_M2)):
                       
                        a_M4[0][x] = M4[x]

                if((len(idx_h_M2) == 1) and (len(idx_w_M2) > 1)): 

                    for x in range(len(idx_w_M2)):
                       
                        a_M4[0][x] = M4[x]

                if((len(idx_h_M2) > 1) and (len(idx_w_M2) > 1)):  

                    k = 0 
                    for x in range(len(idx_h_M2)):
                        for y in range(len(idx_w_M2)):
                           
                            a_M4[x][y] = M4[k]
                            k+=1     

    return [a_M1, a_M2, a_M3, a_M4]  

def qt(img, n): # koder (quad tree)

    ##########################################################################################################
    # Kompresja w oparciu o strukturę danych "quad-tree" - koder() - ALGORYTM REKURENCYJNY :
    # wejście: obraz (gs/rgb), (h,w,d)
    #                                 -> zwraca:
    #                                 pojedynczą zmienną (listę) -> v[] (Zawiera informacje o wymiarach org. obrazu)
    # Struktura danych : lista [],    # -> [ [1,2], [3,4], 5, 6, [[7,8], [9,10]] ]

    #img = [[1]] # Zakładamy, że tak będzie wyglądała forma obrazu wejściowego, który posiada tylko jeden element (ta forma będzie zależeć od tego, w jakiej postaci dane zwróci funkcja split_4)

    # ! jeden element -> : [[1]]
    # WARUNEK : JEDEN ELEMENT -> MUSI MIEĆ TAKĄ FORMĘ : [[1]]
    #    
    # wejście: a[] (lista)
    """ 
        a =
        [[1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]]  
    """
    # zwraca: jedną zmienną -> v[] - (listę) - obraz skompresowany.    
  

    height = len(img) 
    width = len(img[0]) 
   
    # Sprawdzenie czy jest to tylko jeden elment (?)  --->    
    
    if((height == 1) and (width == 1)):

        one_element = True
    else:
        one_element = False

    #print("\n--> one_element ? = ", one_element)

    if(one_element == True):

        #return [img[0][0]] # [img] (?) Czy to jest odpowiednia postać zwróconych danych, w przypadku jednego elementu ? 
        #return img[0][0]

        #if(n==0):
        #    return [img[0][0]]
        #else:
        return img[0][0]

    
    # Sprawdzenie czy wszystkie wartości z listy są takie same (?)  --->        

    first_element = img[0][0]

    the_same = True

    for i in range(height):
        for j in range(width):

            if(first_element != img[i][j]):
                the_same = False
     
    #print("\n--> the_same ? = ", the_same)

    if(the_same == True):
       
        return [img[0][0]]

    ##################################################################################################

    # PODZIAŁ MACIERZY NA 4 CZĘŚCI (!) --->

    M = split_4(img) 

    # [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0]]]
    #  a_M1                       a_M2                      a_M3                     a_M4

    # [[1.0, 1.0], [1.0, 1.0]]  (a_M1)
    # [[0.0, 1.0], [1.0, 0.0]]  (a_M2)  
    # [[1.0, 0.0], [0.0, 1.0]]  (a_M3)     
    # [[1.0, 1.0], [0.0, 1.0]]  (a_M4)   

    v = [] 

    #return [qt(M[0]), qt(M[1]), qt(M[2]), qt(M[3])]

    #print("\nM =  ->\n", M)

    if(n == 0):
        n += 1
        v.append(height)
        v.append(width)
        if(len(M)<4):
            v.append(qt(M[0],n))
            v.append(qt(M[1],n))
        else:
            v.append(qt(M[0],n))
            v.append(qt(M[1],n))
            v.append(qt(M[2],n))
            v.append(qt(M[3],n))       
        
    else:
        #v.append(qt(M[0], 1), qt(M[1], 1), qt(M[2], 1), qt(M[3], 1))
        if(len(M)<4):
            v.append(qt(M[0],n))
            v.append(qt(M[1],n))
        else:
            v.append(qt(M[0],n))
            v.append(qt(M[1],n))
            v.append(qt(M[2],n))
            v.append(qt(M[3],n))   
    
    
    return v    

def decoder_(v, h, w): 
    
    ##########################################################################################################
    # Dekoder - w oparciu o strukturę danych "quad-tree" -  ALGORYTM REKURENCYJNY :
    # wejście: obraz zakodowany - v[] - (gs/rgb), (h,w,d)
    #                                    -> zwraca:
    #                                    pojedynczą zmienną d[] - (listę) -> przechowującą odkodowane wartości obrazu.
    # Struktura danych : lista [],       # -> [ 1,1,1,1,0,0,0,0,1,0,1,1,1,1,0,]

    
    new_w = int(np.ceil((w/2))) 
    new_h = int(np.ceil((h/2)))  

    i = 0

    for ii in range(4):   

        if((isinstance(v[i],int)) == False):

            if(len(v[i]) == 1):               

                for x in range(new_h):
                    for y in range(new_w):
                            
                        d.append(v[i][0])                         
                i+=1

            elif(len(v[i]) != 1):

                # Sprawdzenie, czy liczba elementów == 4, i lista NIE JEST zagnieżdżona           
         
                len_v = len(v[i]) 

                is_v_nested = any(isinstance(i, list) for i in v[i]) 

                if ((len_v == 4) and (is_v_nested == False)): 

                    k = 0

                    for x in range(len(v[i])):         
                     
                            d.append(v[i][k]) 

                            k += 1
                    i+=1                   

                if ((len_v == 4) and (is_v_nested == True)):

                    decoder_(v[i], int(np.ceil((h/2))), int(np.ceil((w/2)))) 

                    i+=1


            else: # jeśli lista nie składa się z jednego, ani 4 elementów ..., to :

                # rekurencja ...
                decoder_(v[i], h, w)              
    
    """
    if((isinstance(v[i],int)) == True):

        if(len([v[i]]) == 1): # BŁĄD ! 

            #print("\n 110 len(v[i]) == 1 \n")

            for x in range(new_h): # 0, 1, 2, 3 ...
                for y in range(new_w): # 0, 1, 2, 3 ...

                    #d.append([v[i][0]]) # z nawiasami czy bez ?
                    d.append(v[i]) 
                    #print("\n 114 d = \n", d)
            i+=1

        #elif(len([v[i]]) != 1):

        #    #print("\n 123 len(v[i]) != 1 \n")

        #    #################################################
        #    # Sprawdzenie, czy liczba elementów == 4, i lista nie jest zagnieżdżona           
 
        #    len_v = len([v[i]]) # długość - liczba elementów == 4.

        #    is_v_nested = any(isinstance(i, list) for i in [v[i]]) # is_v_nested ? (Czy lista jest zagnieżdżona ? 

        #    if ((len_v == 4) and (is_v_nested == False)): # lista ma 4 elementy, i nie jest zagnieżdżona ...

        #        #print("\n (len_v == 4) and (is_v_nested == False) \n")

        #        k = 0

        #        #for x in range(new_h): # 0, 1, 2, 3 ...
        #        #    for y in range(new_w): # 0, 1, 2, 3 ...

        #        #            #d.append([v[i][0]]) # z nawiasami czy bez ?
        #        #        d.append(v[i][k]) 
        #        #        k += 1

        #        for x in range(len([v[i]])): # 0, 1, 2, 3 ...                      

        #                    #d.append([v[i][0]]) # z nawiasami czy bez ?
        #                d.append(v[i][k]) 
        #                k += 1
        #        i+=1
        #        #print("\n 141 d = \n", d)

        #    if ((len_v == 4) and (is_v_nested == True)):
            
        #        #print("\n ... (len_v == 4) and (is_v_nested == True) \n")

        #        #print("\n ... 145d = \n", d)




        #        decoder_(v[i], h/2, w/2) # <-- 

        #        i+=1

        #    #print("\n ... 149 d = \n", d)

        #    # - l
        #    #new_w = int(np.ceil((new_w/2))) # 4
        #    #new_h = int(np.ceil((new_h/2)))   
        #    # TUTAJ POWINNA NASTĄPIĆ REKURENCJA !   (?) DLACZEGO  (?)

        #else: # jeśli lista nie składa się z jednego, ani 4 elementów ..., to :

        #    #print("\n (172) else \n")

        #    # Uruchom dekoder dla danego v[i], :
        #    #print("\n 159 d = \n", d)
        #    decoder_(v[i], h, w)
        #    #print("\n 161 d = \n", d)
    """

print("\n################################################################################\n")
print("\n kompresja strumieniowa ->\n")

orginal_image = plt.imread('test20.png')

print("\n orginal image = \n\n", orginal_image)
print("\n orginal_image dtype = ",orginal_image.dtype)
print("\n type orginal_image  = ", type(orginal_image))
print("\n orginal_image shape = ",orginal_image.shape)

obraz_zakodowany = stream_compression_coder(orginal_image) 

print("\n obraz_zakodowany[0:6] = ", obraz_zakodowany[0:6]) 

print("\n type(obraz_zakodowany)  = ", type(obraz_zakodowany))
print("\n len(obraz_zakodowany) = ", len(obraz_zakodowany)) 

print("\n dekoder -> \n")

obraz_zdekodowany = stream_compression_decoder(obraz_zakodowany) 



if(len(obraz_zakodowany)>orginal_image.shape[0]*orginal_image.shape[1]*orginal_image.shape[2]):
    print("\nlen(obraz_zakodowany) jest większe !\n")
else:
    print("\nobraz jest większy !\n")

print("\n Porównanie rozmiaru  -> ")

v_size = sys.getsizeof(obraz_zakodowany) # v

orginal_size = sys.getsizeof(orginal_image) # rozmiar zajmowanej pamięci przez obiekt w BAJTACH

decoded_size = sys.getsizeof(obraz_zdekodowany) # target

print("\n orginal_size = ", orginal_size, " bytes") # obraz oryginalny 
print("\n v_size = ", v_size, " bytes")             # v - obraz zakodowany (skompresowany)
print("\n decoded_size = ", decoded_size, " bytes") # obraz zdekodowany 

"""

    # Formatowanie tekstu (wartości liczbowych) aby były bardziej czytelne - dodanie spacji co każde 3 znaki : 
    # (na razie pomijam to ...) 
    #locale.setlocale(locale.LC_NUMERIC, 'pl_PL')
    locale.setlocale(locale.LC_ALL, '')
    #orginal_size = format(orginal_size, 'n')
    #target_size = format(target_size, 'n')
    #orginal_size_ = format(orginal_size, 'n')
    #target_size_ = format(target_size, 'n')

"""

print("\n################################################################################\n")

#print("\n -> orginal_size = ", orginal_size, " bytes")
#print("\n -> target_size = ", target_size, " bytes")

v_size = v_size / 1000000 
orginal_size = orginal_size / 1000000 
decoded_size = decoded_size / 1000000 

print("\n -> orginal_size = ", orginal_size, " [MB]")
print("\n -> obraz_zakodowany = ", v_size, " [MB]")
print("\n -> decoded_size = ", decoded_size, " [MB]")

stopien_kompresji = (orginal_size/v_size) # ✓

procent_kompresji = (v_size/orginal_size)*100

"""

    plt.imshow(orginal_image)
        #plt.title("Obraz oryginalny - przed kompresją")
    plt.savefig('_orginal.png')
    plt.show()

    plt.imshow(obraz_zdekodowany)
        #plt.title("Obraz zdekodowany")
    plt.savefig('_decoded.png')
    plt.show()

"""


plt.imshow(orginal_image)
 
x = int(orginal_image.shape[1]*0.55)
y = int(orginal_image.shape[0]*0.9)

print("\n x = ", x)
print("\n y = ", y)


plt.text(x, y, "RLE: stopień kompresji = "+str(stopien_kompresji)+"\n procent kompresji = "+str(procent_kompresji), fontsize=10, color="black", bbox ={'facecolor':'lightblue', 'pad':10})         
plt.title("Obraz oryginalny - przed kompresją")
plt.savefig('_orginal.png')
plt.show()

plt.imshow(obraz_zdekodowany)
plt.title("Obraz zdekodowany")
plt.savefig('_decoded.png')
plt.show()

exit()

plt.imshow(orginal_image)
plt.title("Obraz oryginalny - przed kompresją")
plt.savefig('_orginal.png')
plt.show()

plt.imshow(obraz_zdekodowany)
plt.title("Obraz zdekodowany")
plt.savefig('_decoded.png')
plt.show()


print("\n################################################################################\n")
print("\n quad tree ->\n")

orginal_image = plt.imread('test34.png')

"""
    print("\n orginal image = \n\n", orginal_image)
    print("\n orginal_image dtype = ",orginal_image.dtype)
    print("\n type orginal_image  = ", type(orginal_image))
    print("\n orginal_image shape = ",orginal_image.shape)
"""
R = orginal_image[:,:,0]
G = orginal_image[:,:,1]
B = orginal_image[:,:,2]

Y = 0.299 * R + 0.587 * G + 0.114 * B

print("\n Y = \n", Y)
print("\n type(Y) = ", type(Y))
print("\n Y.dtype = ", Y.dtype)
print("\n Y.shape = ", Y.shape)

print("\n Y[0] ", Y[0])
print("\n Y[0][0] = ", Y[0][0]) 
print("\n len(Y) (height) = ", len(Y))
print("\n len(Y[0]) (width) = ", len(Y[0]))
print("\n Y[0][0] = ", Y[0][0])

img_height = Y.shape[0]
img_width = Y.shape[1]

a = TwoD(img_width, img_height)       
# a ->    [['0', '0', '0', '0'],  
#          ['0', '0', '0', '0']]

# a[0] ->  ['0', '0', '0', '0']    (row)
    # a[0][0] -> 0

# len(a[0]) = width (4)
# len(a) = height (2)

for i in range(img_height):
    for j in range(img_width):     
            
        a[i][j] = Y[i,j]         

if((len(a)>10) and (len(a[0])>10)):
    print("\n type(a) = ", type(a))
    print("\n ---> a = \n")

    print(a[0][0:10])
else:
    print("\n ---> a = \n", a)
    print("\n type(a) = ", type(a))  

#########################################################################################

print("\n################################################################################\n")
print(" -> KODER() | quad tree : \n"z

v = qt(a, 0) # v - jedna zmienna - postać zakodowana

print("\n a -> \n")

if((len(a)>10) and (len(a[0])>10)):
    print(a[0][0:10])
else:
    print(a)

#print("\n v[] -> \n", v)
print("\n v[0] -> \n", v[0])
print("\n v[1] -> \n", v[1])
print("\n v[] -> \n", v[2])
#exit()
print("\n len v[] -> ", len(v))
#exit()
plt.imshow(orginal_image)
plt.show()

#########################################################################################

print("\n################################################################################\n")

print("\n Porównanie rozmiaru  -> ")

v_size = sys.getsizeof(v) # v

orginal_size = sys.getsizeof(orginal_image) # rozmiar zajmowanej pamięci przez obiekt w BAJTACH

#decoded_size = sys.getsizeof(obraz_zdekodowany) # target

print("\n orginal_size = ", orginal_size, " bytes") # obraz oryginalny 
print("\n v_size = ", v_size, " bytes")             # v - obraz zakodowany (skompresowany)



#exit()

"""

    # Formatowanie tekstu (wartości liczbowych) aby były bardziej czytelne - dodanie spacji co każde 3 znaki : 
    # (na razie pomijam to ...) 
    #locale.setlocale(locale.LC_NUMERIC, 'pl_PL')
    locale.setlocale(locale.LC_ALL, '')
    #orginal_size = format(orginal_size, 'n')
    #target_size = format(target_size, 'n')
    #orginal_size_ = format(orginal_size, 'n')
    #target_size_ = format(target_size, 'n')

"""

print("\n################################################################################\n")


v_size = v_size / 1000000 
orginal_size = orginal_size / 1000000 
#decoded_size = decoded_size / 1000000 

print("\n -> orginal_size = ", orginal_size, " [MB]")
print("\n -> obraz_zakodowany (v) = ", v_size, " [MB]")
#print("\n -> decoded_size = ", decoded_size, " [MB]")

#exit()

print("\n################################################################################\n")

print(" -> DEKODER() | quad tree : \n")

print("\nv[] -> \n\n",v)

h = v[0]
w = v[1]

d = [] # potem zmodyfikuj to tak aby było wewnątrz dekodera, i korzystało z rekurencji ...  patrz rozwiązanie w qt() )

v = v[2:]

#print("\n(1183), v[] -> \n\n",v)

print("\n################################################################################\n")

print("\n d = \n", d)

d1 = decoder_(v,img_height,img_width)

print("\n d = \n", d)

exit()