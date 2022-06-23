import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyaudio 
import wave
import soundfile as sf

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 1
RATE = 44100
CHUNK = 1024             # 1024 bytes of data red from a buffer
RECORD_SECONDS = 0.1
filename = "file.wav"

audio = pyaudio.PyAudio()

numdevices = audio.get_device_count()
for i in range(0, numdevices):
        print(audio.get_device_info_by_index(i))

# przechwytywanie jako proces działający w tle, przy wykorzystaniu funkcji callback.
# funkcję która zrealizuje nam ten process bezpośredniego przetwarzania danych:
# Przechowywanie danych  - dane pobrane z mikrofonu - w zmiennej 'in_data'
# Przetwarzanie danych
# Dane zwracana do głośników - w returnie zmiennej 'out_data'.

def process_data(in_data, frame_count, time_info, flag):
    global Frame_buffer,frame_idx
    in_audio_data = np.frombuffer(in_data, dtype=np.int16)
    #Frame_buffer[frame_idx,(frame_idx+CHUNK),0]=in_audio_data
    Frame_buffer[frame_idx:(frame_idx+CHUNK),0]=in_audio_data
    ################################
    ## ! Do something wih data (?)
    out_audio_data = in_audio_data   
    ################################
    Frame_buffer[frame_idx:(frame_idx+CHUNK),1]=out_audio_data
    out_data = out_audio_data.tobytes()   
    frame_idx+=CHUNK     
    return out_data, pyaudio.paContinue

# ! Dane pobrane z mikrofonu będą w zmiennej 'in_data'
# dane przekazywane do waszych głośników -> będą zwracane w returnie w zmiennej 'out_data'
# ! Kolejnym korkiem jest modyfikacja naszego strumienia i przerobienie naszego strumienia wejściowego na wejściowo-wyjściowe :
# należy ustalić odpowiednie urządzenia (indeksy) ! 

stream = audio.open(input_device_index =1, # ?
                    output_device_index=3, # (pc -> 3, 7, 8, 11..)
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=process_data
                    )

# Teraz deklaracja potrzebnych zmiennych globalnych oraz nowa pętla działająca przez 10 sekund 
# (można sterować parametrem):

global Frame_buffer,frame_idx

N=8 # Pętla działająca przez N sekund ...
Frame_buffer = np.zeros(((N+1)*RATE,2))
frame_idx=0

stream.start_stream()
while stream.is_active():
    time.sleep(N)
    stream.stop_stream()
stream.close()

#print("\n Frame_buffer.dtype = \n", type(Frame_buffer))

#########################################################################
# Sprawdzanie danych i zapis do pliku
# Frame_buffer - jest to kanał stereo (2 kanały - lewy, prawy)

#print("\nFrame_buffer = ", Frame_buffer.astype(np.int16), "\n")
print("\nFrame_buffer = ", Frame_buffer, "\n")
print("\nFrame_buffer[:,0] = ", Frame_buffer[:,0], "\n")
print("\nFrame_buffer[:,1] = ", Frame_buffer[:,1], "\n")
print("\nFrame_buffer[:,0].shape = ", Frame_buffer[:,0].shape, "\n")
print("\nFrame_buffer[:,1].shape = ", Frame_buffer[:,1].shape, "\n")
print("\nFrame_buffer.shape = ", Frame_buffer.shape, "\n")
#print("\nFrame_buffer.shape = ", Frame_buffer.shape, "\n")
#print("\nFrame_buffer - np.roll= ", np.roll(Frame_buffer), "\n")

# Parametr - określający czas opóźnienia (w sekundach)
delay_time = 0.2
delay_volume = 0.2

print("\nFrame_buffer = \n", Frame_buffer, "\n")
print("\nFrame_buffer - np.roll (1 sekunda) = /n", np.roll(Frame_buffer,int(delay_time*44100), axis=0), "\n")

Frame_buffer_delay = np.roll(Frame_buffer,int(delay_time*44100), axis=0)

Frame_buffer_result = Frame_buffer + (delay_volume * Frame_buffer_delay)

print("\nFrame_buffer_result = \n", Frame_buffer_result, "\n")


sf.write('a2.wav', Frame_buffer.astype(np.int16), RATE)
sf.write('a2_delay.wav', Frame_buffer_delay.astype(np.int16), RATE) # zapisz z opóźnieniem (przesunięciem - np.roll)
sf.write('a2_result.wav', Frame_buffer_result.astype(np.int16), RATE) # zapisz z opóźnieniem (przesunięciem - np.roll)

## wyświetlanie \

time =  Frame_buffer.shape[0] / RATE # Czas trwania nagrania

time_ = np.linspace(0, Frame_buffer.shape[0] / RATE, num = Frame_buffer.shape[0]) # wektor czasu, zawierający sekundy
                            # 6.9336875    # 332817

plt.subplot(2,1,1)
plt.plot(time_, Frame_buffer[:,0])
plt.title("Frame_buffer[:,0]")
#plt.xlabel("time [s]")

plt.subplot(2,1,2)
plt.plot(time_, Frame_buffer[:,1])
plt.title("Frame_buffer[:,1]")
plt.xlabel("time [s]")

plt.show()

#####################################################
# delay
plt.subplot(3,1,1)
plt.plot(time_, Frame_buffer)
plt.title("Frame_buffer")
#plt.xlabel("time [s]")

plt.subplot(3,1,2)
plt.plot(time_, Frame_buffer_delay)
plt.title("Frame_buffer - delay")
#plt.xlabel("time [s]")

plt.subplot(3,1,3)
plt.plot(time_, Frame_buffer_result)
plt.title("Frame_buffer - result")
plt.xlabel("time [s]")

plt.show()
