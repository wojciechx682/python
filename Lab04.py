import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

import scipy.fftpack
from scipy.interpolate import interp1d

import sys

import math 

from math import sqrt

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
   
    data_q = np.zeros((data.shape[0],data.shape[1]))  

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            sound_value = data[i,j]         

            closest_sound_ = closest_sound(sound_value, sound_palette, j)

            data_q[i,j] = closest_sound_         

    return data_q
    
def quantizaion_(data, data_type):  
    
    if((data.dtype == "float64") or (data.dtype == "float32")):        
        
        data_value_range_min = round(min(min(data[:,0]),min(data[:,1])))
        data_value_range_max = round(max(max(data[:,0]),max(data[:,1])))        

    if((np.issubdtype(data.dtype,np.integer)) == True):   
       
        data_value_range_min =  np.iinfo(data.dtype).min
        data_value_range_max =  np.iinfo(data.dtype).max          

    dtype_value_range_min = np.iinfo(np.str(data_type)).min 
    dtype_value_range_max = np.iinfo(np.str(data_type)).max       
        
    m = data_value_range_min 
    n = data_value_range_max  
        
    data_q = np.zeros((data.shape[0],data.shape[1]))    

    for i in range(data_q.shape[0]):
        for j in range(data_q.shape[1]):            

            F_a = (data[i,j]-m)/(n-m)               

            data_q[i,j] = F_a     

    m = dtype_value_range_min 
    n = dtype_value_range_max      

    for i in range(data_q.shape[0]):
        for j in range(data_q.shape[1]):          

            F_c = round((data_q[i,j]*(n-m))+m)  
           
            data_q[i,j] = F_c    
   
    target = np.zeros((data_q.shape[0], data_q.shape[1]), dtype=data_type)   

    for i in range(data_q.shape[0]):
        for j in range(data_q.shape[1]): 
            
            target[i,j] = round(data_q[i,j])  


    target2 = np.zeros((target.shape[0], target.shape[1]), dtype=np.float32)    

    for i in range(target2.shape[0]):
        for j in range(target2.shape[1]):
            
            target2[i,j] = target[i,j]

    m = np.iinfo(np.str(target.dtype)).min 
    n = np.iinfo(np.str(target.dtype)).max   

    data_q2 = np.zeros((target2.shape[0],target2.shape[1]))  

    for i in range(data_q.shape[0]):
        for j in range(data_q.shape[1]):           

            F_a = (target2[i,j]-m)/(n-m)               

            data_q2[i,j] = F_a      
   
    if((np.issubdtype(data.dtype,np.integer)) == True):
                
         m = np.iinfo(data.dtype).min
         n = np.iinfo(data.dtype).max 
    
    else: 

        m = round(min(min(data[:,0]),min(data[:,1])))
        n = round(max(max(data[:,0]),max(data[:,1])))        

    for i in range(data_q2.shape[0]):
        for j in range(data_q2.shape[1]):            

            F_c = ((data_q2[i,j]*(n-m))+m)   
            
            data_q2[i,j] = F_c

    if((np.issubdtype(data.dtype,np.integer)) == True):        
                
        result = np.zeros((data_q2.shape[0],data_q2.shape[1]), dtype=data.dtype)  

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):

                result[i,j] = int(data_q2[i,j])
                
        return result         

    else:
        
        return data_q2

def decimation(data, n):

    # n - parametr - (liczba ca??kowita) - okre??laj??cy do kt??r?? pr??bk?? uwzgl??dni?? w nowym sygnale (interwa??)

    result = data[::n]

    return result

def decimation_(data, fs, f_new):

    # f_new - okre??na now?? cz??stotliwo???? pr??bkowania

    time = data.shape[0]/fs

    ilosc_probek = time * f_new 

    n = math.ceil(data.shape[0] / ilosc_probek)   

    result = data[::n]

    return result

def resampling(data, f_res, method):   

    time =  data.shape[0] / fs 
    time_ = np.arange(0, data.shape[0]/fs, (data.shape[0]/fs)/data.shape[0]) 

    ilosc_probek = int(f_res * time)

    time_res = np.arange(0, ilosc_probek/f_res, (ilosc_probek/f_res)/ilosc_probek) 

    x1 = time_res

    if (method == "lin"):
        method = interp1d(time_, data[:,0])
        y = method(x1).astype(data[:,0].dtype)

    if (method == "nlin"):        
        method = interp1d(time_, data[:,0], kind='cubic')
        y = method(x1).astype(data[:,0].dtype)
        
    return x1, y  

for k in range(1, 4):  

    for j in range(1, 7):

        data, fs = sf.read("sound/"+str(k)+".wav")        

        print("\n data = \n\n", data)
        print("\n fs = ", fs)
        print("\n data dtype = ", data.dtype)
        print("\n data shape = ", data.shape)

        print("\n data - ilo???? unikalnych warto??ci (L) = ", np.unique(data[:,0]).size)
        print("\n data - ilo???? unikalnych warto??ci (R) = ", np.unique(data[:,1]).size)

        print("\n data 0 min = ", min(data[:,0]))
        print("\n data 0 max = ", max(data[:,0]))

        print("\n data 1 min = ", min(data[:,1]))
        print("\n data 1 max =", max(data[:,1]))
        
        if(j == 1):
            f_new = 125
        if(j == 2):
            f_new = 550
        if(j == 3):
             f_new = 1024 
        if(j == 4):
            f_new = 4000 
        if(j == 5):
            f_new = 16950
        if(j == 6):
            f_new = 24000               
        
        if(k==1):
            k_ = "sin_60 Hz"
        if(k==2):
            k_ = "sin_440 Hz"
        if(k==3):
            k_ = "sin_8000 Hz"

        result_lin = resampling(data, f_new, "lin")
        result_nlin = resampling(data, f_new, "nlin")            

        
        #sf.write('sound/result.wav', data_q, fs)
        
        sf.write("sound/data_r_"+str(k)+"_"+str(j)+"_bit_lin.wav", result_lin[1], f_new)
        sf.write("sound/data_r_"+str(k)+"_"+str(j)+"_bit_nlin.wav", result_nlin[1], f_new)

        #########################################################################################

        time =  data.shape[0] / fs  # Czas trwania utworu

        # Wektor o rozmiarze ILO??CI PR??BEK, od zera do warto??ci CZASU TRWANIA utworu : 

        #print("\n -> data.shape[0] / fs = ", (data.shape[0] / fs) / 10)

        if (k == 1):
            n = 10
        if (k == 2):
            n = 35
        if (k == 3):
            n = 45

       
        time_ = np.linspace(0, data.shape[0] / fs, num = data.shape[0])
       
        print("\n time =", time)  # 6.9336875
        print("\n time_ =", time_) 
        print("\n time_.shape=", time_.shape)                

        plt.subplot(2,2,1)
        plt.plot(time_, data[:,0])
        plt.title("Oryginalny sygna?? audio - "+str(k_)+" - kana?? lewy")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda")       

        plt.subplot(2,2,2)
        plt.plot(time_, data[:,1])
        plt.title("Oryginalny sygna?? audio - "+str(k_)+" - kana?? prawy")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda")        
        
        data_q = result_lin[1].copy()
        data_q1 = result_nlin[1].copy()

        plt.subplot(2,2,3)
        plt.plot(result_lin[0], result_lin[1])
        plt.title("Resampling (metoda liniowa), cz??stotliwos?? pr??bkowania = "+str(f_new)+" [Hz] - Kana?? lewy")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda")
        

        plt.subplot(2,2,4)
        plt.plot(result_nlin[0], result_nlin[1])
        plt.title("Resampling (metoda nieliniowa), cz??stotliwos?? pr??bkowania = "+str(f_new)+" [Hz] - Kana?? lewy")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda")        

        plt.show()

        ###############################################################################        

        print("\n\n Wy??wietlanie widma -> \n\n")        

        print("\n ############################################################################################## \n")       

        fs = f_new

        time =  data_q.shape[0] / fs # Czas trwania utworu (sinus)       

        time_r1 = np.linspace(0, data_q.shape[0] / fs, num = data_q.shape[0])
        time_r2 = np.linspace(0, data_q1.shape[0] / fs, num = data_q1.shape[0])                           

        print("\n time =", time)  
        print("\n time_ = ", time_)
        print("\n time_ shape = ", time_.shape)       

        #########################################################################

        # ANALIZA WIDMOWA - sygna?? sin_440Hz : 

        # wy??wietlenie MODU??U WIDMA : 

        print("\n ############################################################ \n ")

        print("\n\n ANALIZA WIDMOWA - sygna?? sin_"+str(k)+" -> \n\n")

        # Prezentacja modu??u widma dla sinusa 440 Hz bez ??adnych parametr??w i modyfikacji :         

        """

        #plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.arange(0, data_q.shape[0])/fs, data_q[:,0])
        plt.title(str(k_))
        plt.xlabel("t [s]")
        plt.ylabel("Amplituda [dB]")

        #print("\n fs = ", fs)
        #print("\n yf = ", yf)

        plt.subplot(2,1,2)
        yf = scipy.fftpack.fft(data_q[:,0])
        plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
        plt.title(str(k_)+" - Widmo")
        plt.xlabel("f [Hz]")
        plt.ylabel("Amplituda [dB]")
        plt.show()

        print("\n ############################################################ \n ")

        # Zmniejszenie rozmiaru transformaty do 256 : 

        fsize=2**8

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.arange(0, data_q.shape[0])/fs, data_q[:,0])
        plt.title(str(k_))
        plt.xlabel("t [s]")
        plt.ylabel("Amplituda [dB]")

        plt.subplot(2,1,2)
        yf = scipy.fftpack.fft(data_q[:,0],fsize)
        plt.plot(np.arange(0,fs,fs/fsize),np.abs(yf))
        plt.title(str(k_)+" - Widmo - rozmiar zmniejszony do 256")
        plt.xlabel("f [Hz]")
        plt.ylabel("Amplituda [dB]")
        plt.show()

        print("\n ############################################################ \n ")

        # Kolejny etap wy??wietlania modu??u widma to wzi??cie tylko po??owy. Modu?? widma jest symetryczny wzgl??dem cz??stotliwo??ci 0 Hz, wi??c przy jego wy??wietlaniu mo??emy pomin???? jedn?? z jego cz????ci:

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.arange(0,data_q.shape[0])/fs,data_q[:,0])
        plt.title(str(k_)+" - Widmo")
        plt.xlabel("t [s]")
        plt.ylabel("Amplituda [dB]")

        plt.subplot(2,1,2)
        yf = scipy.fftpack.fft(data_q[:,0],fsize)
        plt.plot(np.arange(0,fs/2,fs/fsize),np.abs(yf[:fsize//2]))
        plt.title(str(k_)+" - Widmo")
        plt.xlabel("f [Hz]")
        plt.ylabel("Amplituda [dB]")
        plt.show()

        print("\n ############################################################ \n ")

        """

        # przeskalowanie warto??ci widma do skali decybelowej (dB) : 

        # Prezentacja po??owy modu??u widma dla sinusa 440 Hz ze zmniejszonym do 256 rozmiarem widma wy??wietlona w dB
        
        fsize=2**8


        """
        #widmo w skali normalnej : (nie decybelowej)
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.arange(0,data_q.shape[0])/fs,data_q[:,0])
        plt.title(str(k_)+" - Widmo")
        plt.xlabel("t [s]")
        plt.ylabel("Amplituda")

        plt.subplot(2,1,2)
        yf = scipy.fftpack.fft(data_q[:,0],fsize)
        plt.plot(np.arange(0,fs/2,fs/fsize),np.abs(yf[:fsize//2]))
        plt.title(str(k_)+" - Widmo")
        plt.xlabel("f [Hz]")
        plt.ylabel("Amplituda [dB]")
        plt.show()
        """
               
        #data_q = data_q.copy()        
        #data_q = result_lin[1].copy()
        #data_q1 = result_nlin[1].copy()

        target_ = np.zeros((len(data_q),2))  
        target_n = np.zeros((len(data_q1),2))  

        for i in range(target_.shape[0]):
            for j in range(target_.shape[1]):

                target_[i,j] = data_q[i]
                

        for i in range(target_n.shape[0]):
            for j in range(target_n.shape[1]):

                target_n[i,j] = data_q1[i]               

        print("\n ######################################### = ", )
        print("\n result_lin[1] = ", result_lin[1], " len = ", len(result_lin[1]))
        print("\n result_nlin[1]  = ", result_nlin[1], " len = ", len(result_nlin[1]))

        print("\n data_q = ", data_q)
        print("\n data_q shape= ", data_q.shape)
        print("\n len(data_q))/fs = ", len(data_q)/fs)
        print("\n len(data_q) = ", len(data_q))

        print("\n target_ = \n", target_)
        print("\n target_ shape= ", target_.shape)
        print("\n target_ dtype= ", target_.dtype)   

        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(np.arange(0,target_.shape[0])/fs,target_[:,0])
        plt.xlabel("t [s]")
        plt.ylabel("Amplituda")
        plt.title(str(k_))

        plt.subplot(3,1,2)
        #yf = scipy.fftpack.fft(data_q,fsize)
        yf = scipy.fftpack.fft(target_[:,0],fsize)
        #plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
        plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
        #plt.xlabel("")
        plt.title(str(k_)+" - Widmo - metoda liniowa")
        plt.xlabel("f [Hz]")
        plt.ylabel("Amplituda [dB]")

        plt.subplot(3,1,3)
        #yf = scipy.fftpack.fft(data_q,fsize)
        yf = scipy.fftpack.fft(target_n[:,0],fsize)
        #plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
        plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
        #plt.xlabel("")
        plt.title(str(k_)+" - Widmo - metoda nieliniowa")
        plt.xlabel("f [Hz]")
        plt.ylabel("Amplituda [dB]")

        #########
        #exit()
        #plt.figure()
        #plt.subplot(2,1,1)
        #plt.plot(np.arange(0,data_i.shape[0])/fs_i,data_i[:,0])

        #plt.subplot(2,1,2)
        #yf = scipy.fftpack.fft(data_i[:,0],fsize)
        #plt.plot(np.arange(0,fs_i/2,fs_i/fsize),20*np.log10( np.abs(yf[:fsize//2])))

        plt.show()
        


#########################################################################################

exit()

data_type = "uint8"                    # uint8, int8, int16  # Zmienna okre??laj??ca, docelowy ZAKRES WARTO??CI (< -..., ... > danych pliku audio

data_q_f = quantizaion_(data_f, data_type) # akceptowalne parametry : uint8, int8, int16
data_q_i = quantizaion_(data_i, data_type) # akceptowalne parametry : uint8, int8, int16

print("\n ############################################################################################## \n")
print("\n\n data_f = \n", data_f)
print("\n data_f dtype = ", data_f.dtype)
print("\n data_f shape = ", data_f.shape)

print("\n data_f - ilo???? unikalnych warto??ci (L) = ", np.unique(data_f[:,0]).size)
print("\n data_f - ilo???? unikalnych warto??ci (R) = ", np.unique(data_f[:,1]).size)

print("\n data_f 0 min = ", min(data_f[:,0]))
print("\n data_f 0 max = ", max(data_f[:,0]))

print("\n data_f 1 min = ", min(data_f[:,1]))
print("\n data_f 1 max =", max(data_f[:,1]))

print("\n ############################################################################################## \n")
print("\n\n data_i = \n", data_i)
print("\n data_i dtype = ", data_i.dtype)
print("\n data_i shape = ", data_i.shape)

print("\n data_i - ilo???? unikalnych warto??ci (L) = ", np.unique(data_i[:,0]).size)
print("\n data_i - ilo???? unikalnych warto??ci (R) = ", np.unique(data_i[:,1]).size)

print("\n data_i 0 min = ", min(data_i[:,0]))
print("\n data_i 0 max = ", max(data_i[:,0]))

print("\n data_i 1 min = ", min(data_i[:,1]))
print("\n data_i 1 max =", max(data_i[:,1]))

print("\n ############################################################################################## \n")
print("\n\n data_q_f = \n", data_q_f)
print("\n data_q_f dtype = ", data_q_f.dtype)
print("\n data_q_f shape = ", data_q_f.shape)

print("\n data_q_f - ilo???? unikalnych warto??ci (L) = ", np.unique(data_q_f[:,0]).size)
print("\n data_q_f - ilo???? unikalnych warto??ci (R) = ", np.unique(data_q_f[:,1]).size)

print("\n data_q_f 0 min = ", min(data_q_f[:,0]))
print("\n data_q_f 0 max = ", max(data_q_f[:,0]))

print("\n data_q_f 1 min = ", min(data_q_f[:,1]))
print("\n data_q_f 1 max =", max(data_q_f[:,1]))


print("\n ############################################################################################## \n")
print("\n\n data_q_i = \n", data_q_i)
print("\n data_q_i dtype = ", data_q_i.dtype)
print("\n data_q_i shape = ", data_q_i.shape)

print("\n data_q_i - ilo???? unikalnych warto??ci (L) = ", np.unique(data_q_i[:,0]).size)
print("\n data_q_i - ilo???? unikalnych warto??ci (R) = ", np.unique(data_q_i[:,1]).size)

print("\n data_q_i 0 min = ", min(data_q_i[:,0]))
print("\n data_q_i 0 max = ", max(data_q_i[:,0]))

print("\n data_q_i 1 min = ", min(data_q_i[:,1]))
print("\n data_q_i 1 max =", max(data_q_i[:,1]))


#sf.write("sound/data_q_uint8.wav", data_q.astype('uint8'), fs)
#sf.write("sound/data_q_"+str(data_type)+"_.wav", data_q, fs)
sf.write("sound/float.wav", data_q_f, fs_f)
sf.write("sound/int.wav", data_q_i, fs_i)

fs  = fs_f

print("data len = \n\n", len(data_f)) # ilo???? pr??ek
print("\n fs = \n", fs)     # cz??stotliwo???? pr??bkowania
#print("\n data.shape[0] = \n", data.shape[0]) # 332817

# Aby okre??li?? czas trwania utworu, nale??y podzieli?? ILO???? PR??BEK przez cz??stotliwo???? pr??bkowania (fs)

time =  data_f.shape[0] / fs # Czas trwania utworu

# Wektor o rozmiarze ILO??CI PR??BEK, od zera do warto??ci CZASU TRWANIA utworu.

time_ = np.linspace(0, data_f.shape[0] / fs, num = data_f.shape[0]) # wektor czasu, zawieraj??cy sekundy
                       # 6.9336875    # 332817


print("\n time =", time)  # 6.9336875
print("\n time_ =", time_)
print("\n time_.shape=", time_.shape)


time_2 = np.arange(0, data_f.shape[0] / fs, (data_f.shape[0] / fs)/data_f.shape[0]) # krok =  6.9336875 / 332817  <- odleg??o??c pomi??dzy pr??bkami

print("\ntime_2 =", time_2) 
print("\ntime_2.shape=", time_2.shape)

# Wy??wietlenie sygna??u na wykresie - (osobno dla ka??dego z kana????w) :

plt.subplot(3,2,1)
plt.plot(time_, data_f[:,0], '#1f77b4', alpha=1)
plt.title("Oryginalny sygna?? audio - kana?? lewy")
#plt.plot(time_, data[:,0], label="Kana?? lewy")
#plt.plot(time_, data[:,1], 'r', label="Kana?? prawy")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [dB]")
#plt.savefig('sound/orginal_1.png')
#plt.show()

plt.subplot(3,2,2)
plt.plot(time_, data_f[:,1], '#1f77b4', alpha=1)
plt.title("Oryginalny sygna?? audio - kana?? prawy")
#plt.plot(time_, data[:,0], label="Kana?? lewy")
#plt.plot(time_, data[:,1], 'r', label="Kana?? prawy")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [dB]")
#plt.savefig('sound/orginal_1.png')
#plt.show()

plt.subplot(3,2,3)
plt.plot(time_, data_q_f[:,0], '#1f77b4', alpha=1)
plt.title("Kwantyzacja - typ danych: float - kana?? lewy")
#plt.title("Kwantyzacja do 8 bit??w -> ilo???? warto??ci = 2^8; Format wej??ciowy: float32; Format wyj??ciowy: float32;")
plt.xlabel("czas [s]")
plt.ylabel("amplituda [dB]")
#plt.savefig('sound/float32-uint8.png')
#plt.show()

plt.subplot(3,2,4)
plt.plot(time_, data_q_f[:,1], '#1f77b4', alpha=1)
plt.title("Kwantyzacja - typ danych: float - kana?? prawy")
#plt.title("Kwantyzacja do 8 bit??w -> ilo???? warto??ci = 2^8; Format wej??ciowy: float32; Format wyj??ciowy: float32;")
plt.xlabel("czas [s]")
plt.ylabel("amplituda [dB]")
#plt.savefig('sound/float32-uint8.png')
#plt.show()

plt.subplot(3,2,5)
plt.plot(time_, data_q_i[:,0], '#1f77b4', alpha=1)
plt.title("Kwantyzacja - typ danych: int - kana?? lewy")
#plt.title("Kwantyzacja do 8 bit??w -> ilo???? warto??ci = 2^8; Format wej??ciowy: int32; Format wyj??ciowy: int32;")
plt.xlabel("czas [s]")
plt.ylabel("amplituda [dB]")
#plt.savefig('sound/int32-uint8.png')
#plt.show()

plt.subplot(3,2,6)
plt.plot(time_, data_q_i[:,1], '#1f77b4', alpha=1)
plt.title("Kwantyzacja - typ danych: int - kana?? prawy")
#plt.title("Kwantyzacja do 8 bit??w -> ilo???? warto??ci = 2^8; Format wej??ciowy: int32; Format wyj??ciowy: int32;")
plt.xlabel("czas [s]")
plt.ylabel("amplituda [dB]")
#plt.savefig('sound/int32-uint8.png')
plt.show()

print("\n ############################################################################################## \n")
print("\n ############################################################################################## \n")
print("\n ############################################################################################## \n")

print("\n\n Wy??wietlanie widma -> \n\n")

#data_f, fs_f = sf.read('sound/piano.wav', dtype="float32")  # dane do kwantyzacji metod?? zmiany typu i zakresu warto??ci
data_i, fs_i = sf.read('sound/SM_Lab05/sin_440Hz_2.wav', dtype="int32")

print("\n ############################################################################################## \n")

print("\n data_i = \n\n", data_i)
print("\n fs_i = ", fs_i)
print("\n data_i dtype = ", data_i.dtype)
print("\n data_i shape = ", data_i.shape)

print("\n data_i - ilo???? unikalnych warto??ci (L) = ", np.unique(data_i[:,0]).size)
print("\n data_i - ilo???? unikalnych warto??ci (R) = ", np.unique(data_i[:,1]).size)

print("\n data_i 0 min = ", min(data_i[:,0]))
print("\n data_i 0 max = ", max(data_i[:,0]))

print("\n data_i 1 min = ", min(data_i[:,1]))
print("\n data_i 1 max =", max(data_i[:,1]))

print("\n ############################################################################################## \n")

fs  = fs_i
#print("data len = \n\n", len(data_f)) # ilo???? pr??ek
#print("\n fs = \n", fs)     # cz??stotliwo???? pr??bkowania
#print("\n data.shape[0] = \n", data.shape[0]) # 332817
# Aby okre??li?? czas trwania utworu, nale??y podzieli?? ILO???? PR??BEK przez cz??stotliwo???? pr??bkowania (fs)

time =  data_i.shape[0] / fs # Czas trwania utworu (sinus)

# Wektor o rozmiarze ILO??CI PR??BEK, od zera do warto??ci CZASU TRWANIA utworu.

time_i = np.linspace(0, data_i.shape[0] / fs_i, num = data_i.shape[0])
                        # 6.9336875             # 332817


print("\n time =", time)  # 6.9336875
print("\n time_i = ", time_i)
print("\n time_i shape = ", time_i.shape)

time_2i = np.arange(0, data_i.shape[0] / fs_i, (data_i.shape[0] / fs_i)/data_i.shape[0]) # krok =  6.9336875 / 332817  <- odleg??o??c pomi??dzy pr??bkami


print("\n time_2i =", time_2i) 
print("\n time_2i.shape=", time_2i.shape)

#########################################################################

# ANALIZA WIDMOWA - sygna?? sin_440Hz : 

# wy??wietlenie MODU??U WIDMA : 

print("\n ############################################################ \n ")

print("\n\n ANALIZA WIDMOWA - sygna?? sin_440Hz  -> \n\n")

# Prezentacja modu??u widma dla sinusa 440 Hz bez ??adnych parametr??w i modyfikacji : 

#plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0, data_i.shape[0])/fs_i, data_i[:,0])
plt.title("sin 440 Hz")
#print("\n fs_i = ", fs_i)
#print("\n yf = ", yf)

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data_i[:,0])


plt.plot(np.arange(0,fs_i,1.0*fs_i/(yf.size)),np.abs(yf))
plt.title("sin 440 Hz - Widmo")
plt.show()

print("\n ############################################################ \n ")

# Zmniejszenie rozmiaru transformaty do 256 : 

fsize=2**8

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data_i.shape[0])/fs_i,data_i[:,0])

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data_i[:,0],fsize)
plt.plot(np.arange(0,fs_i,fs_i/fsize),np.abs(yf))
plt.show()

print("\n ############################################################ \n ")

# Kolejny etap wy??wietlania modu??u widma to wzi??cie tylko po??owy. Modu?? widma jest symetryczny wzgl??dem cz??stotliwo??ci 0 Hz, wi??c przy jego wy??wietlaniu mo??emy pomin???? jedn?? z jego cz????ci:

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data_i.shape[0])/fs_i,data_i[:,0])

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data_i[:,0],fsize)
plt.plot(np.arange(0,fs_i/2,fs_i/fsize),np.abs(yf[:fsize//2]))

plt.show()

print("\n ############################################################ \n ")

# przeskalowanie warto??ci widma do skali decybelowej (dB) : 

# Prezentacja po??owy modu??u widma dla sinusa 440 Hz ze zmniejszonym do 256 rozmiarem widma wy??wietlona w dB

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data_i.shape[0])/fs_i,data_i[:,0])

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data_i[:,0],fsize)
plt.plot(np.arange(0,fs_i/2,fs_i/fsize),20*np.log10( np.abs(yf[:fsize//2])))

plt.show()

print("\n ############################################################################################## \n")

print("\n\n Decymacja -> \n\n")

print("\n ############################################################################################## \n")

data = data_f.copy()

print("\n data = \n\n", data, "\n")
print("\n fs = \n\n", fs, "\n")
print("\n data.dtype = ", data.dtype)
print("\n data.shape = ", data.shape)
print("\nnp.unique(data[:,0]).size = ", np.unique(data[:,0]).size)
print("\nnp.unique(data[:,1]).size = ", np.unique(data[:,1]).size)
print("\ndata min value  = ", min(min(data[:,0]),min(data[:,1])))
print("\ndata max value  = ", max(max(data[:,0]),max(data[:,1])))

print("\n ############################################################################################## \n")

result = decimation(data, 12)

time =  data.shape[0] / fs # Czas trwania utworu 

time_ = np.arange(0, data.shape[0]/fs, (data.shape[0]/fs)/data.shape[0]) # time org

fs_dec = round(result.shape[0]/time) # cz??stotliwo???? pr??bkowania sygna??u po decymacji

time_dec = np.arange(0, result.shape[0]/fs_dec, (result.shape[0]/fs_dec)/result.shape[0]) #czas trwania utworu po decymacji 

print("\n time = \n", time)
print("\n time_ = \n", time_)
print("\n len time_ = \n", len(time_))
print("\n shape time_ = \n", time_.shape)

print("\n fs_dec = \n", fs_dec)
print("\n time_dec = \n", time_dec)
print("\n len time_dec = \n", len(time_dec))
print("\n shape time_dec = \n", time_dec.shape)

plt.subplot(2,2,1)
plt.plot(time_, data[:,0])
#plt.plot(time_, data[:,0], label="Kana?? lewy")
#plt.plot(time_, data[:,1], 'r', label="Kana?? prawy")
plt.title("Oryginalny sygna?? audio - kana?? lewy")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [dB]")
#plt.savefig('sound/orginal.png')
#plt.show()

plt.subplot(2,2,2)
plt.plot(time_, data[:,1])
#plt.plot(time_, data[:,0], label="Kana?? lewy")
#plt.plot(time_, data[:,1], 'r', label="Kana?? prawy")
plt.title("Oryginalny sygna?? audio - kana?? prawy")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [dB]")
#plt.savefig('sound/orginal.png')
#plt.show()

plt.subplot(2,2,3)
plt.plot(time_dec, result[:,0])
plt.title("Decymacja - kana?? lewy")
plt.xlabel("czas [s]")
plt.ylabel("amplituda [dB]")
#plt.savefig('sound/dec.png')
#plt.show()

plt.subplot(2,2,4)
plt.plot(time_dec, result[:,1])
plt.title("Decymacja - kana?? prawy")
plt.xlabel("czas [s]")
plt.ylabel("amplituda [dB]")
#plt.savefig('sound/dec.png')

plt.show()

print("\n fs = ", fs)
print("\n fs_dec = ", fs_dec)

sf.write("sound/data_org.wav", data, fs)
sf.write("sound/data_dec.wav", result, fs_dec)

print("\n ############################################################################################## \n")

print("\n\n Resampling, metoda liniowa / nieliniowa -> \n\n")

"""
	czyli, jak ju?? mamy mniej pr??bek, - pr??ba odtworzenia sygna??u		

				(ew. pr??ba dostosowania sygna??u do zupe??nie innej cz??stotliwo??ci pr??bkowania).

	W instrukcji s?? proste algorytmy resamplingu -> metoda liniowa, nieliniowa 
							("s?? w zasadzie takie same" - pod wzgl??dem implementacji, ~ 2 linie kodu).

	Mo??na by by??o spr??bowa?? je por??wna?? - czym si?? od siebie r????ni??.


# Resampling - polega na zmianie cz??stotliwo??ci pr??bkowania.

# upsampling - sztuczne zwi??kszanie cz??stotliwo??ci pr??bkowania,
# downsampling - obni??anie cz??stotliwo??ci pr??bkowania.

# Na zaj??ciach b??dziemy zajmowa?? si?? tylko prostymi metodami ZMNIEJSZANIA cz??stotliwo??ci pr??bkowania

# dwie metody : DECYMACJA oraz INTERPOLACJA 


(3) Resampling - 

	czyli, jak ju?? mamy mniej pr??bek, - pr??ba odtworzenia sygna??u		

				(ew. pr??ba dostosowania sygna??u do zupe??nie innej cz??stotliwo??ci pr??bkowania).
	W instrukcji s?? proste algorytmy resamplingu -> metoda liniowa, nieliniowa 
							("s?? w zasadzie takie same" - pod wzgl??dem implementacji, ~ 2 linie kodu).
	Mo??na by by??o spr??bowa?? je por??wna?? - czym si?? od siebie r????ni??.

Pr??bkowanie

Resampling
"""

####################################################################################################

f_res = 50

result_lin = resampling(data, f_res, "lin")
result_nlin = resampling(data, f_res, "nlin")

#print("\n result_lin -> \n\n", result_lin)
#print("\n result_nlin -> \n\n", result_nlin)


time_ = np.arange(0, data.shape[0]/fs, (data.shape[0]/fs)/data.shape[0]) 

plt.plot(time_, data[:,0])
plt.title("Oryginalny sygna?? audio - kana?? lewy")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [dB]")
plt.savefig('sound/orginal.png')
plt.show()

plt.plot(time_, data[:,1])
plt.title("Oryginalny sygna?? audio - kana?? prawy")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [dB]")
#plt.savefig('sound/orginal.png')
plt.show()


plt.plot(result_lin[0], result_lin[1])
plt.title("Resampling - metoda liniowa")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda [dB]")
plt.savefig('sound/res_lin.png')
plt.show()

plt.plot(result_nlin[0], result_nlin[1])
plt.title("Resampling - metoda nieliniowa")
plt.xlabel("czas [s]")
plt.ylabel("amplituda [dB]")
plt.savefig('sound/res_nlin.png')
plt.show()

exit()

########################################################################################
########################################################################################
########################################################################################