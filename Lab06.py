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

import os
cls = lambda: os.system('cls')
cls()


data, fs = sf.read('sound/sing_low1.wav') 

fn = "sing_low1"

"""

    x = np.linspace(-1,1,1000)
    y = 0.9 * np.sin(np.pi*x*4)

    data = np.zeros((x.shape[0], 2))

    print("\n data = \n", data)

    data[:,0] = y.copy()
    data[:,1] = y.copy()

    fs = 1000

"""

target = np.zeros((len(data),2))

for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        target[i,j] = data[i]

data = target.copy()


def μ_law_coder(data, mu): 
   
    x = data.copy()

    y = np.sign(x)*(np.log(1+(mu*abs(x)))/np.log(1+mu))

    return y
                   
def μ_law_decoder(mu_law_q, mu):            
   
    y = mu_law_q.copy()
   
    x = np.sign(y)*(1/mu)*(((1+mu)**abs(y))-1)

    return x

def A_law_coder(data, A): 
  
    x = data.copy()

    y = np.zeros((x.shape[0],2)) 

    for i in range(x.shape[0]):        
        for j in range(2):        

            if(abs(x[i,j]) < (1/A)):

                y[i,j] = np.sign(x[i,j])*((A*abs(x[i,j]))/(1+np.log(A))) 

            else:
                y[i,j] = np.sign(x[i,j])*((1+np.log(A*abs(x[i,j])))/(1+np.log(A))) 

    return y
   
def A_law_decoder(A_law_q, A):    

    y = A_law_q.copy()
   
    x = np.zeros((y.shape[0],2)) 

    for i in range(y.shape[0]):        
        for j in range(2):        

            if(abs(y[i,j]) < (1/(1+np.log(A)))):

                x[i,j] = np.sign(y[i,j])*((abs(y[i,j])*(1+np.log(A)))/(A))            
                
            else:
                x[i,j] = np.sign(y[i,j])*((np.exp(abs(y[i,j])*(1+np.log(A)-1)))/(A))    
    return x

def generate_sound_palete(data, number_of_values):
    
    data_height = data.shape[0]
    data_width = data.shape[1]       

    L = data[:,0] 
    R = data[:,1]

    sound_values_L = np.zeros((data_height*data_width,1)) 
    sound_values_R = np.zeros((data_height*data_width,1))        

    for i in range(data_height):          

        sound_values_L[i] = L[i]
        sound_values_R[i] = R[i]
        
    sound_palette = np.zeros((min(len(sound_values_L), len(sound_values_R)),2))    

    for i in range(sound_palette.shape[0]):
        for j in range(sound_palette.shape[1]):

            if(j == 0):
                sound_palette[i,j] =  sound_values_L[i]

            if(j == 1):
                sound_palette[i,j] =  sound_values_R[i]           

    sound_palette = np.unique(sound_palette, axis=0)        

    col_1 = np.linspace(min(sound_palette[:,0]), max(sound_palette[:,0]), number_of_values)
    col_2 = np.linspace(min(sound_palette[:,1]), max(sound_palette[:,1]), number_of_values)          
              
    sound_palette = np.zeros((number_of_values,2))  
    
    for i in range(number_of_values):

        for j in range(2):

            if(j == 0):
                sound_palette[i,j] =  col_1[i]

            if(j == 1):
                sound_palette[i,j] =  col_2[i]            

    return sound_palette

def closest_sound(sound_value, sound_palette, w):
                  
    if(sound_palette.shape[1] < 2):    

        c = sound_value
        sound_diffs = []
        for sound in sound_palette:
            cr = sound
            sound_diff = sqrt(abs(c - cr)**2)
            sound_diffs.append((sound_diff, sound))            
        return min(sound_diffs)[1] 


    else:   
        if(w == 0):    
            c = sound_value
            sound_diffs = []
            for sound in sound_palette[:,0]:
                cr = sound
                sound_diff = sqrt(abs(c - cr)**2)
                sound_diffs.append((sound_diff, sound))               
            return min(sound_diffs)[1]
        if(w == 1):    
            c = sound_value
            sound_diffs = []
            for sound in sound_palette[:,1]:
                cg = sound
                sound_diff = sqrt(abs(c - cg)**2)
                sound_diffs.append((sound_diff, sound))         
            return min(sound_diffs)[1]

def quantization(data, sound_palette):     
    
    try:

        data_q = np.zeros((data.shape[0],data.shape[1])) 

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                sound_value = data[i,j]         

                closest_sound_ = closest_sound(sound_value, sound_palette, j)

                data_q[i,j] = closest_sound_         

        return data_q
        
    except:     

        sound_value = data

        closest_sound_ = closest_sound(sound_value, sound_palette, 1)

        data_q = closest_sound_

        return data_q

    """
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):

                sound_value = data[i,j]         

                closest_sound_ = closest_sound(sound_value, sound_palette, j)

                data_q[i,j] = closest_sound_         

        return data_q
    """
   
def dpcm_coder(data, n):

    y = np.zeros((data.shape[0],2))

    sound_palette = generate_sound_palete(data, n)   
    
    E = 0

    e1 = 0
    e2 = 0 

    Y_ = 0

    for i in range(data.shape[0]):       

        e1 = E
        e2 = E

        for j in range(2):             
                      
            if(j == 0):
               
                Y_ = data[i,j] - e1              
                
                Y1 = quantization(Y_, sound_palette)                 
                
                y[i,j] = Y1

            if(j == 1):
               
                Y_ = data[i,j] - e2           
                
                Y2 = quantization(Y_, sound_palette)  
                
                y[i,j] = Y1               

        Y = np.mean([Y1, Y2])   

        E += Y

    return y

def dpcm_decoder(y):

    X = np.zeros((y.shape[0],2))    

    E = 0  
    E1 = 0  
    E2 = 0  

    for i in range(y.shape[0]):     
   
        for j in range(data.shape[1]):  


            if(j==0):
                                                     
                x1 = y[i,j] + E1                                          

                E1 = x1

                X[i,j] = x1    
               
            if(j==1):
                                                
                x2 = y[i,j] + E2                                         

                E2 = x2

                X[i,j] = x2                   

    E = np.mean([E1, E2])           

    return X

    """
        
        E = 0

        e1 = 0
        e2 = 0 

        X_ = 0


        for i in range(data.shape[0]):   # 15 ...               

            e1 = E
            e2 = E

            #for j in range(data.shape[1]):  
            for j in range(2):   # 0, 1             

                    #        #Y_ = data[i,j] - E                                           
                    #Y_ += (data[i,j] - E)

                    
                    #if(j == 1):
                    #    Y_ = Y_/(data.shape[1])            

                    #Y = quantization(Y_, sound_palette)           

                    #E += Y
                    #y[i,j] = Y  
                    ##y[i,j+1] = Y  
                    
                    ##E2 += Y              
                          
                if(j == 0):

                    Y_ = data[i,j] - e1

                    Y1 = quantization(Y_, sound_palette)  
                    #Y1 = quantization_even(Y_)                     
                    
                    y[i,j] = Y1           

                if(j == 1):

                    Y_ = data[i,j] - e2

                    Y2 = quantization(Y_, sound_palette)  
                    #Y2 = quantization_even(Y_)                   

                    y[i,j] = Y1                  


            Y = np.mean([Y1, Y2])

            E += Y

        return y

    """  

class DPCM(object):

    def __init__(self, diff_table):
        self.diff_table = np.array(diff_table)

    def encode(self, wave):
        if len(wave.shape)==2:
            return np.vstack([self.encode(wave[:,0]),self.encode(wave[:,1])]).T
        symbols = np.zeros(len(wave), dtype=np.uint)
        prediction = 0
        for i, model in enumerate(wave):
            predictions = prediction + self.diff_table
            abs_error = np.abs(predictions - model)
            diff_index = np.argmin(abs_error)
            symbols[i] = diff_index
            prediction += self.diff_table[diff_index]
        return symbols

    def decode(self, symbols):
        if len(symbols.shape)==2:
            return np.vstack([self.decode(symbols[:,0]),self.decode(symbols[:,1])]).T
        wave = np.zeros(len(symbols), dtype=np.double)
        prediction = 0
        for i, diff_index in enumerate(symbols):
            prediction += self.diff_table[diff_index]
            wave[i] = prediction
        return wave


print("\n ############################################################################################## \n")

print("\n data = \n\n", data)
print("\n fs = ", fs)
print("\n data dtype = ", data.dtype)
print("\n data shape = ", data.shape)

print("\n data - ilość unikalnych wartości (L) = ", np.unique(data[:,0]).size)
print("\n data - ilość unikalnych wartości (R) = ", np.unique(data[:,1]).size)

print("\n data[:,0] min = ", min(data[:,0]))
print("\n data[:,0] max = ", max(data[:,0]))

print("\n data[:,1] min = ", min(data[:,1]))
print("\n data[:,1] max =", max(data[:,1]))

print("\n ############################################################################################## \n")

mu = 255

mu_law_k = μ_law_coder(data, mu)

print("\n mu_law_k = \n\n", mu_law_k)
print("\n fs = ", fs)
print("\n mu_law dtype = ", mu_law_k.dtype)
print("\n mu_law shape = ", mu_law_k.shape)

print("\n mu_law - ilość unikalnych wartości (L) = ", np.unique(mu_law_k[:,0]).size)
print("\n mu_law - ilość unikalnych wartości (R) = ", np.unique(mu_law_k[:,1]).size)

print("\n mu_law 0 min = ", min(mu_law_k[:,0]))
print("\n mu_law 0 max = ", max(mu_law_k[:,0]))

print("\n mu_law 1 min = ", min(mu_law_k[:,1]))
print("\n mu_law 1 max =", max(mu_law_k[:,1]))

print("\n ############################################################################################## \n")

print("\n KWANTYZACJA -> \n")

bits = 2  # 3 bity ( 8 wartości )          

number_of_values = 2**bits

sound_palette = generate_sound_palete(mu_law_k, number_of_values) # ✓

print("\n sound_palette _ = \n\n", sound_palette) # ✓
print("\n sound_palette.dtype = \n", sound_palette.dtype)
print("\n sound_palette.shape = ", sound_palette.shape)

#########################################################################################
# KWANTYZACJA :      

print("\n###########################################################\n")

# kwantyzacja (mu_law_k) - sygnału skompresowanego (mu_law_k) ->

mu_law_q = quantization(mu_law_k, sound_palette)   

print("\n -> data_q (mu_law_k -> Q -> mu_law_k_q) = \n\n", mu_law_q)
print("\n data_q.dtype = ", mu_law_q.dtype)
print("\n data_q.shape = ", mu_law_q.shape)

print("\n data_q - ilość unikalnych wartości (L) = ", np.unique(mu_law_q[:,0]).size)
print("\n data_q - ilość unikalnych wartości (R) = ", np.unique(mu_law_q[:,1]).size)
print("\n data_q 0 min = ", min(mu_law_q[:,0]))
print("\n data_q 0 max = ", max(mu_law_q[:,0]))

print("\n data_q 1 min = ", min(mu_law_q[:,1]))
print("\n data_q 1 max = ", max(mu_law_q[:,1]))  

print("\n###########################################################\n")

sound_palette = generate_sound_palete(data, number_of_values)

data_q = quantization(data, sound_palette)      

print("\n -> data_q (data) = \n\n", data_q)
print("\n data_q.dtype = ", data_q.dtype)
print("\n data_q.shape = ", data_q.shape)

print("\n data_q - ilość unikalnych wartości (L) = ", np.unique(data_q[:,0]).size)
print("\n data_q - ilość unikalnych wartości (R) = ", np.unique(data_q[:,1]).size)
print("\n data_q 0 min = ", min(data_q[:,0]))
print("\n data_q 0 max = ", max(data_q[:,0]))

print("\n data_q 1 min = ", min(data_q[:,1]))
print("\n data_q 1 max = ", max(data_q[:,1]))


print("\n###########################################################\n")

# mu-law - dekoder() -> 

print("\n mu-law - dekoder() -> \n")

mu_law_d = μ_law_decoder(mu_law_q, mu)      

print("\n -> mu_law_d (data) = \n\n", mu_law_d)
print("\n mu_law_d.dtype = ", mu_law_d.dtype)
print("\n mu_law_d.shape = ", mu_law_d.shape)

print("\n mu_law_d - ilość unikalnych wartości (L) = ", np.unique(mu_law_d[:,0]).size)
print("\n mu_law_d - ilość unikalnych wartości (R) = ", np.unique(mu_law_d[:,1]).size)
print("\n mu_law_d 0 min = ", min(mu_law_d[:,0]))
print("\n mu_law_d 0 max = ", max(mu_law_d[:,0]))

print("\n mu_law_d 1 min = ", min(mu_law_d[:,1]))
print("\n mu_law_d 1 max = ", max(mu_law_d[:,1]))

####################################################################################

time =  data.shape[0] / fs                                     # czas trwania utworu

time_ = np.linspace(0, data.shape[0] / fs, num = data.shape[0]) 

n = 10 # 25

print("\n -> data.shape[0] / fs /"+str(n)+" = ", (data.shape[0] / fs) / n)

time_ = np.linspace(0, (data.shape[0] / fs) / n, num = round(data.shape[0]/n)) 

print("\n bits = ", bits)  
print("\n mu = ", mu)  
print("\n n = ", n) 
print("\n time = ", time)  # 6.9336875
print("\n time_ = ", time_) 
print("\n time_.shape = ", time_.shape)
print("\n round(data.shape[0]/n) = ", round(data.shape[0]/n))

plt.subplot(5,1,1)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
leg = plt.legend(loc='upper right')
plt.title("Liczba bitów kompresji (bits) = "+str(bits)+", mu = "+str(mu)+", n (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

plt.subplot(5,1,2)
plt.plot(time_, mu_law_k[0:round(mu_law_k.shape[0]/n),1], '#1f77b4', alpha=1, label="Kompresja μ-law")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja μ-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/quan_1.png')

plt.subplot(5,1,3)
plt.plot(time_, mu_law_q[0:round(mu_law_q.shape[0]/n),0], '#1f77b4', alpha=1, label="Kompresja μ-law + Kwantyzacja "+str(bits)+" bit")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja mu-law (+ Kwantyzacja) ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

plt.subplot(5,1,4)
plt.plot(time_, data_q[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Sygnał oryginalny - po kwantyzacji "+str(bits)+" bit ")
leg = plt.legend(loc='upper right')
plt.title("Kwantyzacja (data)")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")


plt.subplot(5,1,5)
plt.plot(time_, mu_law_d[0:round(mu_law_d.shape[0]/n),0], '#1f77b4', alpha=1, label="μ_law - sygnał po dekompresji")
leg = plt.legend(loc='upper right')
#plt.title("mu_law - sygnał po dekompresji")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

#plt.show()

sf.write("result/"+str(bits)+"/data.wav", data, fs)
sf.write("result/"+str(bits)+"/mu_law_k_"+str(fn)+"_.wav", mu_law_k, fs)
sf.write("result/"+str(bits)+"/mu_law_q_"+str(fn)+"_.wav", mu_law_q, fs)
sf.write("result/"+str(bits)+"/mu_law_d_"+str(fn)+"_.wav", mu_law_d, fs)
sf.write("result/"+str(bits)+"/data_q_"+str(fn)+"_.wav", data_q, fs)

data_size = sys.getsizeof(data) # data - (rozmiar zajmowanej pamięci przez obiekt w BAJTACH)

mu_law_k_size = sys.getsizeof(mu_law_k) 
mu_law_q_size = sys.getsizeof(mu_law_q) 
data_q_size = sys.getsizeof(data_q) 
mu_law_d_size = sys.getsizeof(mu_law_d) 

print("\n################################################################################\n")


print("\n data_size = ", data_size, " bytes") 
print("\n mu_law_k_size = ", mu_law_k_size, " bytes")            
print("\n mu_law_q_size = ", mu_law_q_size, " bytes")            
print("\n data_q_size = ", data_q_size, " bytes")            
print("\n mu_law_d_size = ", mu_law_d_size, " bytes")          

"""

    print("\n Wartości unikalne -> \n\n")

    print("\n np.unique data = \n\n", np.unique(data, axis=0), "\n ------------------------------------------------------------") 
    print("\n np.unique mu_law_k = \n", np.unique(mu_law_k, axis=0), "\n ------------------------------------------------------------")            
    print("\n np.unique mu_law_k_q = \n", np.unique(mu_law_q, axis=0), "\n ------------------------------------------------------------")            
    print("\n np.unique mu_law_d = \n", np.unique(data_q, axis=0), "\n ------------------------------------------------------------")            
    print("\n np.unique data_q = \n", np.unique(mu_law_d, axis=0), "\n ------------------------------------------------------------") 

"""

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

data_size = data_size / 1000000 
mu_law_k_size = mu_law_k_size / 1000000 

mu_law_q_size = mu_law_q_size / 1000000 
data_q_size = data_q_size / 1000000 
mu_law_d_size = mu_law_d_size / 1000000 

print("\n -> data_size = ", data_size, " [MB]")
print("\n -> mu_law_k_size = ", mu_law_k_size, " [MB]")
print("\n -> mu_law_q_size = ", mu_law_q_size, " [MB]")
print("\n -> data_q_size = ", data_q_size, " [MB]")
print("\n -> mu_law_d_size = ", mu_law_d_size, " [MB]")

stopien_kompresji = (data_size/mu_law_q_size) # ✓

print("\n -> stopien_kompresji = ", stopien_kompresji, "\n")

plt.show()

####################################################################################
# KOMPRESJA A-LAW

print("\n ############################################################################################## \n")

A = 87.6

A_law_k = A_law_coder(data, A)

print("\n A_law_k = \n\n", A_law_k)
print("\n fs = ", fs)
print("\n A_law_k dtype = ", A_law_k.dtype)
print("\n A_law_k shape = ", A_law_k.shape)

print("\n A_law_k - ilość unikalnych wartości (L) = ", np.unique(A_law_k[:,0]).size)
print("\n A_law_k - ilość unikalnych wartości (R) = ", np.unique(A_law_k[:,1]).size)

print("\n A_law_k 0 min = ", min(A_law_k[:,0]))
print("\n A_law_k 0 max = ", max(A_law_k[:,0]))

print("\n A_law_k 1 min = ", min(A_law_k[:,1]))
print("\n A_law_k 1 max =", max(A_law_k[:,1]))

print("\n###########################################################\n")

# kwantyzacja (A_law_k) - sygnału skompresowanego (A_law_k) ->

sound_palette = generate_sound_palete(A_law_k, number_of_values)

A_law_q = quantization(A_law_k, sound_palette)   

print("\n -> A_law_q (A_law_k -> Q -> A_law_k_q) = \n\n", A_law_q)
print("\n A_law_q.dtype = ", A_law_q.dtype)
print("\n A_law_q.shape = ", A_law_q.shape)

print("\n A_law_q - ilość unikalnych wartości (L) = ", np.unique(A_law_q[:,0]).size)
print("\n A_law_q - ilość unikalnych wartości (R) = ", np.unique(A_law_q[:,1]).size)
print("\n A_law_q 0 min = ", min(A_law_q[:,0]))
print("\n A_law_q 0 max = ", max(A_law_q[:,0]))

print("\n A_law_q 1 min = ", min(A_law_q[:,1]))
print("\n A_law_q 1 max = ", max(A_law_q[:,1]))  

print("\n###########################################################\n")

# Dekompresja (A_law_q) - sygnału skwantyzowanego (A_law_q) ->

A_law_d = A_law_decoder(A_law_q, A)   

print("\n -> A_law_d = \n\n", A_law_d)
print("\n A_law_d.dtype = ", A_law_d.dtype)
print("\n A_law_d.shape = ", A_law_d.shape)

print("\n A_law_d - ilość unikalnych wartości (L) = ", np.unique(A_law_d[:,0]).size)
print("\n A_law_d - ilość unikalnych wartości (R) = ", np.unique(A_law_d[:,1]).size)
print("\n A_law_d 0 min = ", min(A_law_d[:,0]))
print("\n A_law_d 0 max = ", max(A_law_d[:,0]))

print("\n A_law_d 1 min = ", min(A_law_d[:,1]))
print("\n A_law_d 1 max = ", max(A_law_d[:,1]))  

print("\n################################################################################\n")


print("\n -> data.shape[0] / fs /"+str(n)+" = ", (data.shape[0] / fs) / n)

#time_ = np.linspace(0, (data.shape[0] / fs) / n, num = round(data.shape[0]/n)) 

print("\n bits = ", bits)  
print("\n A = ", A)  
print("\n n = ", n) 
print("\n time = ", time)  # 6.9336875
print("\n time_ = ", time_) 
print("\n time_.shape = ", time_.shape)
print("\n round(data.shape[0]/n) = ", round(data.shape[0]/n))

plt.subplot(5,1,1)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
leg = plt.legend(loc='upper right')
plt.title("Liczba bitów kompresji (bits) = "+str(bits)+", mu = "+str(mu)+", n (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

plt.subplot(5,1,2)
plt.plot(time_, A_law_k[0:round(A_law_k.shape[0]/n),1], '#1f77b4', alpha=1, label="Kompresja A-law")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja μ-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/quan_1.png')

plt.subplot(5,1,3)
plt.plot(time_, A_law_q[0:round(A_law_q.shape[0]/n),0], '#1f77b4', alpha=1, label="Kompresja A-law + Kwantyzacja "+str(bits)+" bit")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja mu-law (+ Kwantyzacja) ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

plt.subplot(5,1,4)
plt.plot(time_, data_q[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Sygnał oryginalny - po kwantyzacji "+str(bits)+" bit ")
leg = plt.legend(loc='upper right')
plt.title("Kwantyzacja (data)")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")


plt.subplot(5,1,5)
plt.plot(time_, A_law_d[0:round(A_law_d.shape[0]/n),0], '#1f77b4', alpha=1, label="A_law - sygnał po dekompresji")
leg = plt.legend(loc='upper right')
#plt.title("mu_law - sygnał po dekompresji")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

#plt.show()

#sf.write("result/data.wav", data, fs)
sf.write("result/"+str(bits)+"/A_law_k_"+str(fn)+"_.wav", A_law_k, fs)
sf.write("result/"+str(bits)+"/A_law_q_"+str(fn)+"_.wav", A_law_q, fs)
sf.write("result/"+str(bits)+"/A_law_d_"+str(fn)+"_.wav", A_law_d, fs)
#sf.write("result/data_q_"+str(bits)+"_.wav", data_q, fs)

data_size = sys.getsizeof(data) # data - (rozmiar zajmowanej pamięci przez obiekt w BAJTACH)

A_law_k_size = sys.getsizeof(A_law_k) 
A_law_q_size = sys.getsizeof(A_law_q) 
data_q_size = sys.getsizeof(data_q) 
A_law_d_size = sys.getsizeof(A_law_d) 

print("\n################################################################################\n")

#print("\n -> orginal_size = ", orginal_size, " bytes")
#print("\n -> target_size = ", target_size, " bytes")

data_size = data_size / 1000000 
mu_law_k_size = mu_law_k_size / 1000000 

mu_law_q_size = mu_law_q_size / 1000000 
data_q_size = data_q_size / 1000000 
mu_law_d_size = mu_law_d_size / 1000000 


print("\n -> data_size = ", data_size, " [MB]")
print("\n -> mu_law_k_size = ", mu_law_k_size, " [MB]")
print("\n -> mu_law_q_size = ", mu_law_q_size, " [MB]")
print("\n -> data_q_size = ", data_q_size, " [MB]")
print("\n -> mu_law_d_size = ", mu_law_d_size, " [MB]")

stopien_kompresji = (data_size/mu_law_q_size) # ✓

print("\n -> stopien_kompresji = ", stopien_kompresji, "\n")

plt.show()
#exit()

plt.subplot(3,1,1)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
plt.plot(time_, mu_law_k[0:round(mu_law_k.shape[0]/n),0], '#1f77b4', alpha=1, label="mu-law k ")
plt.plot(time_, A_law_k[0:round(A_law_k.shape[0]/n),0], '#3feb6a', alpha=1, label="A-law k")
leg = plt.legend(loc='upper right')
plt.title("Liczba bitów kompresji (bits) = "+str(bits)+", mu = "+str(mu)+", A = "+str(A)+", n (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

plt.subplot(3,1,2)
plt.plot(time_, mu_law_q[0:round(mu_law_q.shape[0]/n),0], '#1f77b4', alpha=1, label="mu-law q ")
plt.plot(time_, A_law_q[0:round(A_law_q.shape[0]/n),0], '#3feb6a', alpha=1, label="A-law q ")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja μ-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/quan_1.png')

plt.subplot(3,1,3)
plt.plot(time_, mu_law_d[0:round(mu_law_d.shape[0]/n),0], '#1f77b4', alpha=1, label="mu-law d ")
plt.plot(time_, A_law_d[0:round(A_law_d.shape[0]/n),0], '#3feb6a', alpha=1, label="A-law d ")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja mu-law (+ Kwantyzacja) ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

plt.show()
#exit()
print("\n################################################################################\n")

print("\n################################################################################\n")

plt.subplot(5,1,1)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
leg = plt.legend(loc='upper right')
plt.title("bits = "+str(bits)+", A = "+str(A)+", n - (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

plt.subplot(5,1,2)
plt.plot(time_, A_law_k[0:round(A_law_k.shape[0]/n),1], '#1f77b4', alpha=1, label="Kompresja A-law")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja μ-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/quan_1.png')

plt.subplot(5,1,3)
plt.plot(time_, A_law_q[0:round(A_law_q.shape[0]/n),0], '#1f77b4', alpha=1, label="Kompresja A-law + Kwantyzacja "+str(bits)+" bit")
leg = plt.legend(loc='upper right')
#plt.title("Kompresja mu-law (+ Kwantyzacja) ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

plt.subplot(5,1,4)
plt.plot(time_, data_q[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="data - sygnał po kwantyzacji "+str(bits)+" bit ")
leg = plt.legend(loc='upper right')
plt.title("Kwantyzacja (data)")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

plt.subplot(5,1,5)
plt.plot(time_, A_law_d[0:round(A_law_d.shape[0]/n),0], '#1f77b4', alpha=1, label="A_law - sygnał po dekompresji")
leg = plt.legend(loc='upper right')
#plt.title("mu_law - sygnał po dekompresji")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")

#plt.show()

#sf.write("sound/data.wav", data, fs)
#sf.write("sound/"+str(bits)+"/A_law_k.wav", A_law_k, fs)
#sf.write("sound/"+str(bits)+"/A_law_k_q.wav", A_law_q, fs)
#sf.write("sound/"+str(bits)+"/A_law_d.wav", A_law_d, fs)
#sf.write("sound/"+str(bits)+"/data_q.wav", data_q, fs)

data_size = sys.getsizeof(data) # data - (rozmiar zajmowanej pamięci przez obiekt w BAJTACH)

A_law_k_size = sys.getsizeof(A_law_k) 
A_law_q_size = sys.getsizeof(A_law_q) 
data_q_size = sys.getsizeof(data_q) 
A_law_d_size = sys.getsizeof(A_law_d) 

print("\n################################################################################\n")

print("\n data_size = ", data_size, " bytes") 
print("\n A_law_k_size = ", A_law_k_size, " bytes")            
print("\n A_law_q_size = ", A_law_q_size, " bytes")            
print("\n data_q_size = ", data_q_size, " bytes")            
print("\n A_law_d_size = ", A_law_d_size, " bytes")   

print("\n################################################################################\n")

print("\n################################################################################\n")
print("\n################################################################################\n")
print("\n################################################################################\n")
print("\n################################################################################\n")

print("\n Wykres - wartość sygnału wejściowego (x), Wartość sygnału wyjściowego (y) -> \n")

#x = np.linspace(min(data[:,0]), max(data[:,0]), fs)
#x = np.linspace(-1, 1, fs)

x = np.sort(data[:,0])
y = np.sort(mu_law_k[:,0])

print("\n x -> ", x)
print("\n y -> ", y)

plt.subplot(1,2,1)
plt.plot(x, y, '#458cff', alpha=1, label="Sygnał po kompresji μ-law - bez kwantyzacji")
plt.xlabel("Wartość sygnału wejściowego")
plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper center')

y2 = np.sort(mu_law_q[:,0])

plt.subplot(1,2,2)
plt.plot(x, y2, '#458cff', alpha=1, label="Sygnał po kompresji μ-law - po kwantyzacji do "+str(bits)+" bitów")
plt.xlabel("Wartość sygnału wejściowego")
plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper center')

print("\n data[:,0] shape -> ", data.shape[0])
print("\n mu_law_d[:,0] shape -> ", data_q.shape[0])

print("\n data[:,0] -> ", data[:,0])
print("\n mu_law_d[:,0] -> ", data_q[:,0])

plt.show()

x = np.sort(data[:,0])
y = np.sort(A_law_k[:,0])

print("\n x -> ", x)
print("\n y -> ", y)
plt.subplot(1,2,1)
plt.plot(x, y, '#458cff', alpha=1, label="Sygnał po kompresji A-law - bez kwantyzacji")
plt.xlabel("Wartość sygnału wejściowego")
plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper left')

y2 = np.sort(A_law_q[:,0])
plt.subplot(1,2,2)
plt.plot(x, y2, '#458cff', alpha=1, label="Sygnał po kompresji A-law - po kwantyzacji do "+str(bits)+" bitów")
plt.xlabel("Wartość sygnału wejściowego")
plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper left')

plt.show()

print("\n###########################################################################\n")

print("\n -> \n")

x = np.sort(data[:,0])

y = np.sort(A_law_q[:,0])

y1 = np.sort(A_law_k[:,0])

y2 = np.sort(mu_law_q[:,0])

y3 = np.sort(mu_law_k[:,0])

plt.subplot(1,2,1)
#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, y, '#1f77b4', alpha=1, label="Sygnał po kompresji A-law - po kwantyzacji do "+str(bits)+" bitów")
leg = plt.legend(loc='upper left')
plt.title("Krzywa kompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")


#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, y1, '#f0b13c', alpha=1, label="Sygnał po kompresji A-law - bez kwantyzacji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper left')
#plt.title("Krzywa kompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, y2, '#3feb6a', alpha=1, label="Sygnał po kompresji μ-law - po kwantyzacji do "+str(bits)+" bitów")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper left')
#plt.title("Krzywa kompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, y3, '#f03c3c', alpha=1, label="Sygnał po kompresji μ-law - bez kwantyzacji")
plt.xlabel("Wartość sygnału wejściowego")
plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper left')
#plt.title("Krzywa kompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

print("\n 847 \n")
#####################################################################################################

y = np.sort(A_law_q[:,0])

y1 = np.sort(A_law_d[:,0])

y2 = np.sort(mu_law_d[:,0])

y3 = np.sort(data_q[:,0])


plt.subplot(1,2,2)
#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, x, '#1f77b4', alpha=1, label="Sygnał oryginalny")

leg = plt.legend(loc='upper left')
plt.title("Krzywa po dekompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")


#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, y1, '#f0b13c', alpha=1, label="Sygnał po dekompresji z A-law - (kwantyzacja "+str(bits)+" bit)")
leg = plt.legend(loc='upper left')
#plt.title("Krzywa kompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, y2, '#3feb6a', alpha=1, label="Sygnał po dekompresji z μ-law - (kwantyzacja "+str(bits)+" bit)")
leg = plt.legend(loc='upper left')
#plt.title("Krzywa kompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

#plt.plot(time_, data[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(x, y3, '#f03c3c', alpha=1, label="Sygnał oryginalny po kwantyzacji do "+str(bits)+" bitów")
plt.xlabel("Wartość sygnału wejściowego")
plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper left')
#plt.title("Krzywa kompresji")
#plt.xlabel("Wartość sygnału wejściowego")
#plt.ylabel("Wartość sygnału wyjściowego")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

plt.show()

print("\n###########################################################################\n")

print("\n DPCM -> \n")

sound_palette = generate_sound_palete(data, number_of_values)

dpcm_k = dpcm_coder(data, number_of_values)

print("\n dpcm_k -> \n", dpcm_k)
print("\n dpcm_k.shape -> \n", dpcm_k.shape)
print("\n dpcm_k.dtype -> \n", dpcm_k.dtype)

x = np.sort(data[:,0])
y = np.sort(dpcm_k[:,0])

print("\n x -> ", x)
print("\n y -> ", y)

plt.subplot(1,2,1)
#plt.plot(x, y, '#458cff', alpha=1, label="Sygnał po kompresji DPCM (po kwantyzacji do "+str(bits)+" bitów")
plt.plot(x, y, '#458cff', alpha=1, label="Sygnał po kompresji DPCM (po kwantyzacji do "+str(bits)+" bitów)")
plt.xlabel("Wartość sygnału wejściowego")
plt.ylabel("Wartość sygnału wyjściowego")
leg = plt.legend(loc='upper right')

time =  data.shape[0] / fs                                

time_ = np.linspace(0, data.shape[0] / fs, num = data.shape[0]) 

plt.subplot(1,2,2)
plt.plot(time_, dpcm_k[:,0], '#1f77b4', alpha=1)
plt.title("DPCM - sygnał po kompresji + kwantyzacji "+str(bits)+" bitów")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
plt.show()

print("\n###########################################################################\n")

print("\n dpcm (dekoder) -> \n")

dpcm_d = dpcm_decoder(dpcm_k)      

print("\n -> dpcm_d (dpcm_d) = \n\n", dpcm_d)
print("\n dpcm_d.dtype = ", dpcm_d.dtype)
print("\n dpcm_d.shape = ", dpcm_d.shape)

print("\n dpcm_d - ilość unikalnych wartości (L) = ", np.unique(dpcm_d[:,0]).size)
print("\n dpcm_d - ilość unikalnych wartości (R) = ", np.unique(dpcm_d[:,1]).size)
print("\n dpcm_d 0 min = ", min(dpcm_d[:,0]))
print("\n dpcm_d 0 max = ", max(dpcm_d[:,0]))

print("\n dpcm_d 1 min = ", min(dpcm_d[:,1]))
print("\n dpcm_d 1 max = ", max(dpcm_d[:,1]))

#dpcm_decoder

######################################################################

plt.plot(time_, data[:,0], '#1f77b4', alpha=1, label="Sygał oryginalny")
#plt.title("Sygał oryginalny")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału (float -> <-1,1>)")
leg = plt.legend(loc='upper left')


plt.plot(time_, A_law_d[:,0], '#de943a', alpha=1, label="Sygnał po dekompresji z A-law")
#plt.title("Sygnał po dekompresji z A-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
leg = plt.legend(loc='upper left')


plt.plot(time_, mu_law_d[:,0], '#41db3b', alpha=1, label="Sygnał po dekompresji z mu-law")
#plt.title("Sygnał po dekompresji z mu-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
leg = plt.legend(loc='upper left')


plt.plot(time_, dpcm_d[:,0], '#e33030', alpha=1, label="Sygnał po dekompresji z DPCM")
#plt.title("Sygnał po dekompresji z DPCM")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału ")
leg = plt.legend(loc='upper left')

#sf.write("result/data.wav", data, fs)
#sf.write("result/dpcm_k"+str(bits)+"_.wav", dpcm_k, fs)
#sf.write("result/dpcm_d"+str(bits)+"_.wav", dpcm_d, fs)
sf.write("result/"+str(bits)+"/dpcm_k"+str(fn)+"_.wav", dpcm_k, fs)
sf.write("result/"+str(bits)+"/dpcm_d"+str(fn)+"_.wav", dpcm_d, fs)

#sf.write("result/data_q_"+str(bits)+"_.wav", data_q, fs)

plt.show()


print("\n ############################################################################################## \n")

print("\n -> Porównanie wszystkich metod (mu-law, A-law, DPCM : \n\n")

plt.subplot(4,1,1)
plt.title("Porównanie wszystkich metod")
plt.plot(time_, data[:,0], '#1f77b4', alpha=1, label="Sygał oryginalny")
#plt.title("Sygał oryginalny")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
leg = plt.legend(loc='upper left')


plt.subplot(4,1,2)
#plt.plot(time_, A_law_d[:,0], '#de943a', alpha=1, label="Sygnał po dekompresji z A-law (kompresja -> kwantyzacja -> dekompresja)")
plt.plot(time_, A_law_d[:,0], '#de943a', alpha=1, label="Sygnał po dekompresji z A-law")
#plt.title("Sygnał po dekompresji z A-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
leg = plt.legend(loc='upper left')

plt.subplot(4,1,3)
plt.plot(time_, mu_law_d[:,0], '#41db3b', alpha=1, label="Sygnał po dekompresji z mu-law")
#plt.title("Sygnał po dekompresji z mu-law")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
leg = plt.legend(loc='upper left')

plt.subplot(4,1,4)
plt.plot(time_, dpcm_d[:,0], '#e33030', alpha=1, label="Sygnał po dekompresji z DPCM")
#plt.title("Sygnał po dekompresji z DPCM")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
leg = plt.legend(loc='upper left')


plt.show()

# Example usage:

fs = 1000

#wave = 128*np.cos(2*np.pi*np.arange(fs)/fs)

x = np.linspace(-1,1,fs)
wave = 0.9 * np.sin(np.pi*x*4)

#x = np.linspace(-1,1,fs)
#y = 0.9 * np.sin(np.pi*x*4)

data = np.zeros((x.shape[0], 2))

#print("\n x = \n", x)
#print("\n x.dtype = \n", x.dtype)
#print("\n x.shape = \n", x.shape)

#print("\n y = \n", y)
#print("\n y.dtype = \n", y.dtype)
#print("\n y.shape = \n", y.shape)

data[:,0] = wave.copy() # (wave == y)
data[:,1] = wave.copy()

print("\n data = \n", data)


print("\n###########################################################\n")

# KWANTYZACJA ->


bits = 2
number_of_values = 2 ** bits

sound_palette = generate_sound_palete(data, number_of_values)

data_q = quantization(data, sound_palette)      

print("\n -> data_q (data) = \n\n", data_q)
print("\n data_q.dtype = ", data_q.dtype)
print("\n data_q.shape = ", data_q.shape)

print("\n data_q - ilość unikalnych wartości (L) = ", np.unique(data_q[:,0]).size)
print("\n data_q - ilość unikalnych wartości (R) = ", np.unique(data_q[:,1]).size)
print("\n data_q 0 min = ", min(data_q[:,0]))
print("\n data_q 0 max = ", max(data_q[:,0]))

print("\n data_q 1 min = ", min(data_q[:,1]))
print("\n data_q 1 max = ", max(data_q[:,1]))


print("\n###########################################################\n")

#y = 0.9 * np.sin(np.pi*x*4)

#wave = np.cos(2*np.pi*np.arange(fs)/fs)

#print("\n wave \n= ", wave)
#print("\n wave.shape = ", wave.shape)
#print("\n wave.dtype = ",  wave.dtype)

wave = np.vstack([wave,wave]).T

time =  wave.shape[0] / fs                                     # czas trwania utworu
time_ = np.linspace(0, wave.shape[0] / fs, num = wave.shape[0]) 

print("\n wave = \n", wave)
print("\n wave.shape = ", wave.shape)
print("\n wave.dtype = ",  wave.dtype)

plt.subplot(4,1,1)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
#plt.plot(time_, wave[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(time_, wave[:,0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio (wave)")
leg = plt.legend(loc='upper right')
#plt.title("Liczba bitów kompresji (bits) = "+str(bits)+", mu = "+str(mu)+", n (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

#plt.show()
#exit()

#half_table = np.array([1,2,4,8,16,32,64,128])

#half_table = np.array([1,2,4,8,16,32,64,128])
max_value_in_signal = max(max(wave[:,0]), max(wave[:,1]))
print("\n max_value_in_signal = \n", max_value_in_signal)

# wartosć max, ilosć wartości (numer of values - zależy od bitó kwantyzacji)

print("\n number_of_values = ", number_of_values)

half_table = np.linspace(0,max_value_in_signal,number_of_values)

#exit()


print("\n half_table = \n", half_table)
print("\n half_table.shape = ", half_table.shape)
print("\n half_table.dtype = ",  half_table.dtype)



diff_table = np.hstack([half_table,-half_table])

print("\n diff_table = \n", diff_table)
print("\n diff_table.shape = ", diff_table.shape)
print("\n diff_table.dtype = ",  diff_table.dtype)



codec = DPCM(diff_table)

print("\n codec = \n", codec) 
# to jest obiekt
#  <__main__.DPCM object at 0x000001D7845D7D00>



dpcm_wave = codec.decode(codec.encode(wave))
dpcm_wave_k = codec.encode(wave)


plt.subplot(4,1,2)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
#plt.plot(time_, wave[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(time_, dpcm_wave_k[:,0], '#f0b13c', alpha=1, label="dpcm_wave - kompresja")
leg = plt.legend(loc='upper right')
#plt.title("Liczba bitów kompresji (bits) = "+str(bits)+", mu = "+str(mu)+", n (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")



plt.subplot(4,1,3)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
#plt.plot(time_, wave[0:round(data.shape[0]/n),0], '#1f77b4', alpha=1, label="Oryginalny sygnał audio")
plt.plot(time_, dpcm_wave[:,0], '#f0b13c', alpha=1, label="dpcm_wave - dekompresja")
leg = plt.legend(loc='upper right')
#plt.title("Liczba bitów kompresji (bits) = "+str(bits)+", mu = "+str(mu)+", n (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

plt.subplot(4,1,4)
#plt.plot(time_, data[:,0], '#1f77b4', alpha=1)
plt.plot(time_, data_q[:,0], '#cf33f2', alpha=1, label="data_q")
leg = plt.legend(loc='upper right')
#plt.title("Liczba bitów kompresji (bits) = "+str(bits)+", mu = "+str(mu)+", n (współczynnik powiększenia wykresu - plot) = "+str(n)+" ")
plt.xlabel("Czas [s]")
plt.ylabel("Wartość sygału")
#plt.savefig('sound/orginal_1.png')
#plt.plot(time_, data[:,0], label="Kanał lewy")

print("\n dpcm_wave = \n", dpcm_wave) 

plt.show()
exit()