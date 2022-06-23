import cv2
import numpy as np

#cap = cv2.VideoCapture(0)            # przechwytywanie z kamery o id -> 0
cap = cv2.VideoCapture("video480.mp4") # przechwytywanie z pliku

# program służący do wyświetlania obrazu : 
# q - wyłączenie programu

"""if not cap.isOpened():
    print("Connot open camera") # or Error reading video file
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Cant receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()"""

# DOMYŚLNY ROZMIAR KLATKI możemy ODCZYTAĆ lub ZMIENIĆ za pomocą odpowiqednich poleceń set oraz get:

#cap.get(cv2.CAP_PROP_FRAME_WIDTH) # width
#cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # height
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280) # set width
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720) # set height

####################################################################################################
# ZAPISYWANIE VIDEO DO PLIKU
# W jaki sposób zapisać nasz strumień wideo do pliku?
# - funkcja zapisu, kodek

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
print("\nZapisywanie video do pliku : \n")

if not cap.isOpened():
    print("Error reading video file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # width -> int
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # height -> int
fps = cap.get(cv2.CAP_PROP_FPS) # do sprawdzenia na potem

# Here you can define your croping values
x,y,h,w = 100,100,350,350

print("frame_width = ", frame_width)
print("frame_height = ", frame_height)

#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))

size = (frame_width, frame_height)  

fourcc = cv2.VideoWriter_fourcc(*'MJPG') # DIVX, XVID, MJPG, X264
                                                                              
#out = cv2.VideoWriter('out.avi', fourcc, 20.0, size)
    #out = cv2.VideoWriter('output.avi', fourcc, 10, (640, 480))

# output
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_mod = cv2.VideoWriter('modified.avi', fourcc, fps, size)
out_org = cv2.VideoWriter('original.avi', fourcc, fps, size)
# https://stackoverflow.com/questions/61723675/crop-a-video-in-python

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Cant receive frame (stream end?). Exiting ...")
        break         

    # Obróć każdą klatkę -> 0 - obrót wg OX, 1 - obrót wg OY, -1 -> OX, OY
    # cv2.flip() - zwracana wartość: obraz
    
    #frame = frame[250:500,250:500].copy()

    #crop_frame = frame[250:500,250:500].copy()
    crop_frame = frame[y:y+h, x:x+w]

    crop_frame = cv2.flip(crop_frame,0)     
    
    # Zmiana rozmiaru wyciętego fragmentu -> (854 x 480)

    r_width = 854
    r_height = 480
    crop_frame = cv2.resize(crop_frame,(r_width,r_height))


    # Write the frame into the
    # file '....avi'
    out_org.write(frame)
    out_mod.write(crop_frame)    

    # Display the frame
    # saved in the file
    cv2.imshow('Frame', frame)    
    cv2.imshow('croped',crop_frame)

    # Press S on keyboard 
    # to stop the process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
    # Break the loop
    #else:
    #    break

# When everything done, release 
# the video capture and video 
# write objects

cap.release()
out_org.release()
out_mod.release()
cv2.destroyAllWindows()

print("The video was successfully saved")

"""print("\n frame data type = \n", type(frame))
print("\n frame shape = \n", frame.shape)
print("\n ret = \n", ret)
print("\n frame = \n", frame)"""

"""out = cv2.VideoWriter('out.avi', fourcc, 20.0, size)
#out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
"""
"""
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
# !!!! (?)
# Potem wewnątrz pętli należy przekazać klatkę do zapisu :
while True:
    out.write(frame)
    out.release()
# A na końcu pamiętać, żeby zamknąć strumień wyjściowy.
# """














































