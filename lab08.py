import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
cls = lambda: os.system('cls')
cls()

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat = '.\\wideo'                          # katalog z plikami wideo
plik = "clip_1.mp4"                       # nazwa pliku
ile = 15                            # ile klatek odtworzyc? <0 - calosc
key_frame_counter = 7                     # co ktora klatka ma byc kluczowa i nie podlegac kompresji
plot_frames = np.array([5.10])             # automatycznie wyrysuj wykresy
auto_pause_frames = np.array([25])        # automatycznie zapauzuj dla klatki i wywietl wykres
subsampling = "4:1:0"                     # parametry dla chorma subsamplingu
wyswietlaj_kaltki = True                  # czy program ma wyswietlac kolejene klatki
comp_type = 1

##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:

    def init(self):

        self.Y = None
        self.Cb = None
        self.Cr = None

def chroma_subsampling(YCrCb, type):
   
    if(type == "4:4:4"):

        return YCrCb

    if(type == "4:2:2"): # 4:2:2 - czyli redukujemy co drugą kolumnę

        for w in range(2): # Cr Cb
            for i in range(YCrCb.shape[0]): # height
                for j in range(YCrCb.shape[1]): # width

                    if((j % 2) != 0): # niepatzyste kolumny
                    
                        if(w == 0): #(range = 2)

                            YCrCb[i,j,1] = YCrCb[i,j-1,1] # Cr

                        if(w == 1):
                            
                            YCrCb[i,j,2] = YCrCb[i,j-1,2] # Cb
        return YCrCb

    if(type == "4:4:0"): # 4:4:0 - czyli redukujemy co drugi wiersz

        for w in range(2): # Cr Cb
            for i in range(YCrCb.shape[0]): # height
                for j in range(YCrCb.shape[1]): # width

                    if((i % 2) != 0): # niepatzyste wiersze
                    
                        if(w == 0): #(range = 2)

                            YCrCb[i,j,1] = YCrCb[i-1,j,1] # Cr

                        if(w == 1):
                            
                            YCrCb[i,j,2] = YCrCb[i-1,j,2] # Cb

        return YCrCb

    # 4:2:0 - czyli redukujemy co drugą kolumnę i co drugi wiersz:

    if(type == "4:2:0"): 

        for w in range(2): # Cr Cb
            for i in range(YCrCb.shape[0]): # height
                for j in range(YCrCb.shape[1]): # width

                    if( ((i % 2) != 0) and ((j % 2) != 0) ): # niepatzyste wiersze
                    
                        if(w == 0): #(range = 2)

                            YCrCb[i,j,1] = YCrCb[i-1,j-1,1] # Cr

                        if(w == 1):
                            
                            YCrCb[i,j,2] = YCrCb[i-1,j-1,2] # Cb
        return YCrCb

    # 4:1:1 - czyli redukujemy wszytkie kolumny poza pierwszą (zostawiamy co 4 kolumnę):

    if(type == "4:1:1"): 

        for w in range(2): # Cr Cb
            for i in range(YCrCb.shape[0]): # height
                for j in range(YCrCb.shape[1]): # width

                    if((j % 4) != 0): # kolumna -> 1, 2, 3,     5, 6, 7 ...
                    
                        if(w == 0): #(range = 2)                            

                            j_floor = int(np.floor(j/4))

                            indeks_kolumny = j_floor * 4 

                            YCrCb[i,j,1] = YCrCb[i,indeks_kolumny,1] # Cr


                        if(w == 1):
                            
                            j_floor = int(np.floor(j/4))

                            indeks_kolumny = j_floor * 4 

                            YCrCb[i,j,2] = YCrCb[i,indeks_kolumny,2] # Cb
        return YCrCb

    # 4:1:0 - czyli wszystkie wartości poza pierszą wartością w pierwszym wierszu (pomijamy co drugi wiesz i zostawiamy co 4-tą kolumnę):

    if(type == "4:1:0"): 

        for w in range(2): # Cr Cb
            for i in range(YCrCb.shape[0]): # height
                for j in range(YCrCb.shape[1]): # width

                    
                    if( ((i % 2) != 0)): # niepatzyste wiersze -> 1, 3, 5
                    
                        if((j % 4) != 0): # kolumna -> 1, 2, 3,     5, 6, 7 ...
                    
                            if(w == 0): #(range = 2)                            

                                j_floor = int(np.floor(j/4))

                                indeks_kolumny = j_floor * 4 

                                YCrCb[i,j,1] = YCrCb[i-1,indeks_kolumny,1] # Cr

                            if(w == 1):
                                
                                j_floor = int(np.floor(j/4))

                                indeks_kolumny = j_floor * 4 

                                YCrCb[i,j,2] = YCrCb[i-1,indeks_kolumny,2] # Cb

                    if((j % 4) != 0): # kolumna -> 1, 2, 3,     5, 6, 7 ...
                    
                        if(w == 0): #(range = 2)                            

                            j_floor = int(np.floor(j/4))

                            indeks_kolumny = j_floor * 4 

                            YCrCb[i,j,1] = YCrCb[i,indeks_kolumny,1] # Cr

                        if(w == 1):
                            
                            j_floor = int(np.floor(j/4))

                            indeks_kolumny = j_floor * 4 

                            YCrCb[i,j,2] = YCrCb[i,indeks_kolumny,2] # Cb
        return YCrCb                        
 
def ThreeD(a, b, c):

    ##########################################################################################################
    # Funkcja tworząca listę trójwymiarową (i,j,w) - (3D):

        lst = [[['0' for col in range(a)] for col in range(b)] for row in range(c)]
        return lst  

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

        for i in range(img_height):
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

        for i in range(len(a)): 
            for j in range(len(a[0])): 
                for w in range(len(a[0][0])): 

                    aa.append(a[i][j][w])   

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
    img_n_dim = v[2]
    
    target = ThreeD(img_n_dim, img_width, img_height)   

    v = v[3:]
   
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
            for w in range(img_n_dim):        

                target[i][j][w] = aa[k]

                k+=1      

    return target  

def compress(Y, Cb, Cr, key_frame_Y, key_frame_Cb, key_frame_Cr, sub_type, comp_type):
                        # KF         # KF          # KF
    data.Y = Y   # (0)
    data.Cr = Cr # (1)
    data.Cb = Cb # (2)
    
    #####################################################################
    # (1) Chroma Subsampling

    #YCrCb = np.dstack([data.Y, data.Cr, data.Cb]).astype(np.uint8)
    YCrCb = np.dstack([data.Y, data.Cr, data.Cb])

    if(sub_type == "4:4:4"): 

        YCrCb_s = chroma_subsampling(YCrCb, "4:4:4")
    if(sub_type == "4:2:2"): 

        YCrCb_s = chroma_subsampling(YCrCb, "4:2:2")

    if(sub_type == "4:4:0"): 

        YCrCb_s = chroma_subsampling(YCrCb, "4:4:0")

    if(sub_type == "4:2:0"): 

        YCrCb_s = chroma_subsampling(YCrCb, "4:2:0")

    if(sub_type == "4:1:1"): # 4:1:0

        YCrCb_s = chroma_subsampling(YCrCb, "4:1:1")

    if(sub_type == "4:1:0"): # 4:1:0

        YCrCb_s = chroma_subsampling(YCrCb, "4:1:0")

    
    data.Y  = YCrCb_s[:,:,0]    # (0)    
    data.Cr = YCrCb_s[:,:,1]    # (1)
    data.Cb = YCrCb_s[:,:,2]    # (2)

    a = YCrCb_s[:,:,0]
    b = YCrCb_s[:,:,1]
    c = YCrCb_s[:,:,2]

    #####################################################################

    #####################################################################
    # (2) Kodowanie różnic (R)

    R_Y  = key_frame_Y - data.Y
    R_Cr = key_frame_Cr - data.Cr
    R_Cb = key_frame_Cb - data.Cb

    if(comp_type == 2):

        R_Y  = R_Y/2
        R_Cr = R_Cr/2
        R_Cb = R_Cb/2

    if(comp_type==4):

        R_Y  = R_Y/4
        R_Cr = R_Cr/4
        R_Cb = R_Cb/4

    if(comp_type==8):

        R_Y  = R_Y/8
        R_Cr = R_Cr/8
        R_Cb = R_Cb/8

    #print("\n R_Y = \n", R_Y)
    #print("\n R_Y.shape = \n", R_Y.shape)
    #print("\n R_Cr = \n", R_Cr)
    #print("\n R_Cb = \n", R_Cb)

    #####################################################################
    # (3) Kompresja RLE

    data_R = np.dstack([R_Y, R_Cr, R_Cb]) # (R)

    #print("\n -> data_R = \n", data_R)    
    #print("\n -> data_R = \n", data_R.shape) # (720, 1280, 3)
    #print("\n -> data_R type = \n", type(data_R)) # (720, 1280, 3)
    #print("\n -> data_R dtype = \n", data_R.dtype) # (720, 1280, 3)
    
    data_RLE = stream_compression_coder(data_R)

    #print("\n -> data_RLE = \n", data_RLE[0:10])  
    #print("\n -> type(data_RLE) = \n", type(data_RLE))  
    #print("\n -> len(data_RLE) = \n", len(data_RLE))  

    
    #print("\n -> len(data_RLE) = \n", len(data_RLE))  

    #data.Y = R_Y   # (0)
    #data.Cr = R_Cr # (1)
    #data.Cb = R_Cb # (2)

    #return data
    return data_RLE, R_Y, R_Cr, R_Cb # data_RLE -> lista, (len = 4846819)

def decompress(data, key_frame_Y, key_frame_Cb, key_frame_Cr , comp_type):

    data_RLE_d = stream_compression_decoder(data)

    #print("\n 357 -> data_RLE_d = \n", data_RLE_d[0])  
    #print("\n -> type(data_RLE_d) = \n", type(data_RLE_d))  
    #print("\n -> len(data_RLE_d) = \n", len(data_RLE_d))

    #exit()

    data_R_h = data[0]
    data_R_w = data[1]
    data_R_d = data[2]

    #print("data_R_h = ", data_R_h)
    #print("data_R_w = ", data_R_w)
    #print("data_R_d = ", data_R_d)

    data_R = np.zeros((data_R_h, data_R_w, data_R_d), dtype="uint8")

    #print("data_R = ", data_R)
    #print("data_R shape = ", data_R.shape)
    #print("data_R dtype = ", data_R.dtype)
    #print("data_R type = ", type(data_R)) 
    
    for i in range(data_R_h):
        for j in range(data_R_w):
            for w in range(data_R_d):

                data_R[i,j,w] = data_RLE_d[i][j][w]

    #for i in range(img_height):
    #    for j in range(img_width): 
    #        for w in range(img_num_of_dim):
    #           
    #            a[i][j][w] = orginal_image[i,j,w] 

    #print(" \n 403 -> data_R = \n", data_R)
    #print("data_R shape = ", data_R.shape)
    #print("data_R dtype = ", data_R.dtype)
    #print("data_R type = ", type(data_R))
    
    #exit()





    data_Y  = data_R[:,:,0]
    data_Cr = data_R[:,:,1]
    data_Cb = data_R[:,:,2]

    #frame = np.dstack([data.Y, data.Cr, data.Cb]).astype(np.uint8)

    #print(" 1 -> \n", key_frame_Y)
    #print(" 2 -> \n", key_frame_Cr)
    #print(" 3 -> \n", key_frame_Cb)

    #print(" 4 type  -> \n", type(data[:,:,0]))
    #print(" 4 -> \n", data[:,:,0])
    #print(" 5 -> \n", data[:,:,1])
    #print(" 6 -> \n", data[:,:,2])

    if(comp_type == 2):

        R_Y_O  = key_frame_Y - (2*data_Y)
        R_Cr_O = key_frame_Cr - (2*data_Cr)
        R_Cb_O = key_frame_Cb - (2*data_Cb)

    if(comp_type == 4):

        R_Y_O  = key_frame_Y - (4*data_Y)
        R_Cr_O = key_frame_Cr - (4*data_Cr)
        R_Cb_O = key_frame_Cb - (4*data_Cb)

    if(comp_type == 8):

        R_Y_O  = key_frame_Y - (8*data_Y)
        R_Cr_O = key_frame_Cr - (8*data_Cr)
        R_Cb_O = key_frame_Cb - (8*data_Cb)

    else:

        R_Y_O  = key_frame_Y - data_Y
        R_Cr_O = key_frame_Cr - data_Cr
        R_Cb_O = key_frame_Cb - data_Cb

    #data.R_Y_2 = data.R_Y/2
    #data.R_Cr_2 = data.R_Cr/2
    #data.R_Cb_2 = data.R_Cb/2

    #data.R_Y_4 = data.R_Y/4
    #data.R_Cr_4 = data.R_Cr/4
    #data.R_Cb_4 = data.R_Cb/4

    frame = np.dstack([R_Y_O, R_Cr_O, R_Cb_O]).astype(np.uint8)

    return frame

def compare_size(frame_Y, frame_Cr, frame_Cb, frame_cY, frame_cCr, frame_cCb, with_RLE, sub_type):

    # frame_Y - warstwa Y przed kompresją.
    # frame_Cr - warstwa Y przed kompresją.
    # frame_Cb - warstwa Y przed kompresją.

    # frame_cY - skompresowana klatka (po Ch. Sub.)

    if(with_RLE == "True"):

        # dla RLE ...
        if(isinstance(frame_cY, list)): # czyli to nie jest klatka kluczowa, jest to lista

            before_RLE_size = frame_Y.shape[0] * frame_Y.shape[1]

            after_rle_size_cY = len(frame_cY)
            after_rle_size_cCr = len(frame_cCr)
            after_rle_size_cCb = len(frame_cCb)

        else: # to jest klatka kluczowa

            before_RLE_size = frame_Y.shape[0] * frame_Y.shape[1]

            print("\n frame_cY = \n", frame_cY)
            print("\n frame_cY = \n", type(frame_cY))
            print("\n frame_cY = \n", frame_cY.dtype)
            print("\n frame_cY -> = \n", len(frame_cY.shape))
            print("\n frame_cY = \n", frame_cY.size)

            after_rle_size_cY = len(stream_compression_coder(frame_cY))
            after_rle_size_cCr = len(stream_compression_coder(frame_cCr))
            after_rle_size_cCb = len(stream_compression_coder(frame_cCb))


        st_Y  = 1-((after_rle_size_cY)/(before_RLE_size))
        st_Cr = 1-((after_rle_size_cCr)/(before_RLE_size))
        st_Cb = 1-((after_rle_size_cCb)/(before_RLE_size))

        return st_Y, st_Cr, st_Cb

    else:

        # bez RLE ->

        """
            f_Y  = np.unique(np.reshape(frame_Y,(frame_Y.shape[0]*frame_Y.shape[1],)))
            f_Cr = np.unique(np.reshape(frame_Cr,(frame_Cr.shape[0]*frame_Cr.shape[1],)))
            f_Cb = np.unique(np.reshape(frame_Cb,(frame_Cb.shape[0]*frame_Cb.shape[1],)))

            print("\n 447 -> \n")

            print("\n f_Y = \n", f_Y)
            print("\n f_Y = \n", f_Y.shape)

            print("\n f_Cr = \n", f_Cr)
            print("\n f_Cr = \n", f_Cr.shape)

            print("\n f_Cb = \n", f_Cb)
            print("\n f_Cb = \n", f_Cb.shape)

            f_cY  = np.unique(np.reshape(frame_cY,(frame_cY.shape[0]*frame_cY.shape[1],)))
            f_cCr = np.unique(np.reshape(frame_cCr,(frame_cCr.shape[0]*frame_cCr.shape[1],)))
            f_cCb = np.unique(np.reshape(frame_cCb,(frame_cCb.shape[0]*frame_cCb.shape[1],)))

            print("\n f_cY = \n", f_cY)
            print("\n f_cY = \n", f_cY.shape)

            print("\n f_cCr = \n", f_cCr)
            print("\n f_cCr = \n", f_cCr.shape)

            print("\n f_cCb = \n", f_cCb)
            print("\n f_cCb = \n", f_cCb.shape)

            # Stopień chrominancji : 

            st_Y = ((f_Y.shape[0]/f_cY.shape[0])-1)*100
            st_Cr = ((f_Cr.shape[0]/f_cCr.shape[0])-1)*100
            st_Cb = ((f_Cb.shape[0]/f_cCb.shape[0])-1)*100

            print("\n st_Y = ", st_Y)
            print("\n st_Cr = ", st_Cr)
            print("\n st_Cb = ", st_Cb)
        """

        if(sub_type == "4:4:4"):
        
            ch_sub_M_size = frame_Y.shape[0]*frame_Y.shape[1]

        if((sub_type == "4:2:2") or (sub_type == "4:4:0")):
        
            ch_sub_M_size = 32

        if((sub_type == "4:2:0") or (sub_type == "4:1:1")):
        
            ch_sub_M_size = 16

        if((sub_type == "4:1:0")):
        
            ch_sub_M_size = 8


        #st_Y  = ((frame_Y.shape[0]*frame_Y.shape[1])-ch_sub_M_size)/(frame_Y.shape[0]*frame_Y.shape[1])
        #st_Cr = ((frame_Cr.shape[0]*frame_Cr.shape[1])-ch_sub_M_size)/(frame_Y.shape[0]*frame_Y.shape[1])
        #st_Cb = ((frame_Cb.shape[0]*frame_Cb.shape[1])-ch_sub_M_size)/(frame_Y.shape[0]*frame_Y.shape[1])

        #ompression_information[0,i] = (frame[:,:,0].size - cY.size)/frame[:,:,0].size                          
        #compression_information[1,i] = (frame[:,:,0].size - cCb.size)/frame[:,:,0].size                              
        #compression_information[2,i] = (frame[:,:,0].size - cCr.size)/frame[SW:,:,0].size

        #st_Y  = ((frame_Y.shape[0]*frame_Y.shape[1])-ch_sub_M_size)/(frame_Y.shape[0]*frame_Y.shape[1])
        #st_Cr = ch_sub_M_size/((frame_Cr.shape[0]*frame_Cr.shape[1]))
        #st_Cb = ch_sub_M_size/((frame_Cr.shape[0]*frame_Cr.shape[1]))

        after_size = ((frame_Cr.shape[0]*frame_Cr.shape[1]))/ch_sub_M_size

        #st_Y  = 0
        #st_Cr = (1/(1-((after_size)/(frame_Cr.shape[0]*frame_Cr.shape[1])) ))-1
        #st_Cb = (1/(1-((after_size)/(frame_Cr.shape[0]*frame_Cr.shape[1]))))-1

        st_Y  = 0
        st_Cr = 1/(1-((after_size)/(frame_Cr.shape[0]*frame_Cr.shape[1]) ) )-1
        st_Cb = 1/(1-((after_size)/(frame_Cr.shape[0]*frame_Cr.shape[1]) ) )-1

        #return f_Y.shape[0], f_Cr.shape[0], f_Cb.shape[0], f_cY.shape[0], f_cCr.shape[0], f_cCb.shape[0]
        return st_Y, st_Cr, st_Cb


##############################################################################
####     Główna petla programu      ##########################################
##############################################################################

cap = cv2.VideoCapture(kat+'\\'+plik) # Przechwycenie wideo

if ile<0: #15 
    ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Normal Frame')
cv2.namedWindow('Decompressed Frame')

compression_information = np.zeros((3,ile)) # 15

x = 0

for i in range(ile): # Główna pętla programu (Po długości klatek ...)

    ret, frame = cap.read() # frame - zawiera klatki, ret (?)

    if wyswietlaj_kaltki: # czy wyświetlamy klatki filmu na ekranie ?

        cv2.imshow('Normal Frame', frame)

    # POBIERAMY NOWĄ KLATKĘ i konwertujemy ją do -> YCrCb 

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    if (i % key_frame_counter) == 0: # jeśli jest to klatka kluczowa

        # (pobieranie klatek kluczowych)

        key_frame = frame # key_frame - klatka kluczowa -> (720, 1280, 3)

        cY = frame[:,:,0]   # Y   # Tu masz warstwy Oryginalnego obrazu (Y, Cr, Cb)
        cCr = frame[:,:,1]  # Cr
        cCb = frame[:,:,2]  # Cb

        d_frame = frame   # d_frame

    else: # kompresja
        #print("\n458 ############################################")
        #lista                         # Y            # Cb          # Cr          # KF              # KF              # KF
        cdata, cY, cCr, cCb = compress(frame[:,:,0], frame[:,:,2], frame[:,:,1], key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], subsampling, comp_type)
                           
            #cY = cdata.Y    # Y - warstwa skompresowana -> Ch_Sub, (R)
            #cCr = cdata.Cr
            #cCb = cdata.Cb        

        # cdata - to jest LISTA (RLE), (len = 4846819)

        d_frame = decompress(cdata, key_frame[:,:,0], key_frame[:,:,2], key_frame[:,:,1], 4)
    
        cdata_size = sys.getsizeof(cdata)

        #print("\n cdata size = ", cY_size)
        #after_size = sys.getsizeof(cdata)


    """   
    if (i % key_frame_counter) != 0:

        print("\n 477 cY = \n", cY)
        print("\n cCr = \n", cCr)
        print("\n cCb = \n", cCb)

        cY_size = sys.getsizeof(np.unique(cY))
        cCr_size = sys.getsizeof(np.unique(cCr))
        cCb_size = sys.getsizeof(np.unique(cCb))
        #######################################################################################

        before_size_Y  = sys.getsizeof(np.unique(key_frame[:,:,0]))
        before_size_Cr  = sys.getsizeof(np.unique(key_frame[:,:,1]))
        before_size_Cb  = sys.getsizeof(np.unique(key_frame[:,:,2]))

        print("\n frame[:,:,0] = \n", key_frame[:,:,0])
        print("\n frame[:,:,1] = \n", key_frame[:,:,1])
        print("\n frame[:,:,2] = \n", key_frame[:,:,2])

        print("\n before_size_Y = \n", before_size_Y)
        print("\n before_size_Cr = \n", before_size_Cr)
        print("\n before_size_Cb = \n", before_size_Cb)

        print("\n 494 cY_size = \n", cY_size)
        print("\n cCr_size = \n", cY_size)
        print("\n cCb_size = \n", cY_size)
    """




     # v

    #print("\n orginal_size = ", orginal_size)
    #print("\n v_size = ", v_size)

    #exit()
                           # Y
    #compression_information[0,i] = (frame[:,:,0].size)/d_frame[:,:,0].size
    
    #data_RLE_size = 

    com_Y, com_Cr, com_Cb = compare_size(frame[:,:,0], frame[:,:,1], frame[:,:,2], cY, cCr, cCb, "True", subsampling)

    compression_information[0,i] = com_Y                 
    compression_information[1,i] = com_Cr                        
    compression_information[2,i] = com_Cb

    """

    if (i % key_frame_counter) != 0: 

        compression_information[0,x] = before_size_Y/cY_size
                                   # Cr
        #compression_information[1,x] = (frame[:,:,0].size - cCr.size)/d_frame[:,:,0].size    
        compression_information[1,x] = before_size_Cr/cCr_size
                                   # Cb
        #compression_information[2,x] = (frame[:,:,0].size - cCb.size)/d_frame[:,:,0].size
        compression_information[2,x] = before_size_Cb/cCb_size

        x = x + 1

    """
    #######################################################################################

    #x = x + 1
    if wyswietlaj_kaltki:

        cv2.imshow('Decompressed Frame', cv2.cvtColor(d_frame, cv2.COLOR_YCrCb2BGR))
    
    """if np.any(plot_frames==i): # rysuj wykresy

        # bardzo słaaby i sztuczny przyklad wykrozystania tej opcji

        fig, axs = plt.subplots(1, 3, sharey=True)
        fig.set_size_inches(16,5)
        axs[0].imshow(frame)
        axs[2].imshow(d_frame) 
        diff = frame.astype(float)-d_frame.astype(float)
        print(np.min(diff), np.max(diff))
        axs[1].imshow(diff,vmin=np.min(diff),vmax=np.max(diff))
    """
    if np.any(auto_pause_frames==i):
        cv2.waitKey(-1) # wait until any key is pressed
    
    k = cv2.waitKey(1) & 0xff
    
    if k==ord('q'):
        break

    elif k == ord('p'):        
        cv2.waitKey(-1) # wait until any key is pressed

#plt.figure()
#plt.plot(np.arange(0,ile),compression_information[0,:]*100)
#plt.plot(np.arange(0,ile),compression_information[1,:]*100)
#plt.plot(np.arange(0,ile),compression_information[2,:]*100)

plt.figure()

#plt.subplot(1,3,1)
plt.plot(np.arange(0,ile), compression_information[0,:], label="Y")
leg = plt.legend(loc='center right')

#plt.subplot(1,3,2)
plt.plot(np.arange(0,ile), compression_information[1,:], label="Cr")
leg = plt.legend(loc='center right')

#plt.subplot(1,3,3)
plt.plot(np.arange(0,ile), compression_information[2,:], label="Cb")
leg = plt.legend(loc='center right')

plt.title("plik: "+str(plik)+", subsampling: " + str(subsampling) + " bez RLE")
plt.xlabel("Numer klatki")
plt.ylabel("% zysku pamięci")
plt.show()

print("\n compression_information = \n", compression_information)
print("\n compression_information[0,:] = \n", compression_information[0,:])
print("\n compression_information[1,:] = \n", compression_information[1,:])
print("\n compression_information[2,:]= \n", compression_information[2,:])

print("\n compression_information shape = \n", compression_information.shape)

print("\n frame = \n", frame)
print("\n frame.shape = \n", frame.shape)

print("\n ret = \n", ret) # ret = True

print("\n cY = \n", cY) # ret = True

print("\n użycie mojej funkcji -> \n\n")


#compare_size(frame[:,:,0], frame[:,:,1], frame[:,:,2], cY, cCr, cCb, "False")

#if(len(obraz_zakodowany)>orginal_image.shape[0]*orginal_image.shape[1]*orginal_image.shape[2]):
#    print("\nlen(obraz_zakodowany) jest większe !\n")
#else:
#    print("\nobraz jest większy !\n")