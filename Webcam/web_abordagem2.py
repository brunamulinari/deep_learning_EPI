# -*- coding: utf-8 -*-
import tflite_runtime.interpreter as tflite
import platform
import cv2
import numpy as np
import io
from threading import Thread
import time


# Inicializa nomes dos labels
labels = ['sem capacete', 'com capacete']

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True



# Carrega o modelo e adquire informações das entradas
# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='Sample_TFLite_models/modelo2.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(1280,720),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    
    img = frame
    
    alt, larg, _ = img.shape
    img_cvt = (img.astype('float32')/127.5)-1 # Limites entre -1 e 1
    img_cvt = cv2.resize(img_cvt, (320, 320)) # Ajusta o tamanho p/ entrada do modelo (320x320)
    img_cvt = cv2.cvtColor(img_cvt, cv2.COLOR_BGR2RGB) # Ajusta p/ formato de entrada (RGB)
    # img_cvt = np.expand_dims(img_cvt, axis=0) # Corrige p/ o tensor ([x, y, c] -> [1, x, y, c])
    
    # Prepara p/ Quantização
    es, zp = input_details[0]['quantization']
    img_cvt = img_cvt / es + zp
    img_cvt = img_cvt.astype(input_details[0]['dtype'])

    img_cvt = np.expand_dims(img_cvt, axis=0) # Corrige p/ o tensor ([x, y, c] -> [1, x, y, c])
    
    # Apresenta imagem p/ o modelo
    interpreter.set_tensor(input_details[0]['index'], img_cvt)
    interpreter.invoke()

    # Extrai os resultados
    output_details = interpreter.get_output_details()
    
    
    # Boxes
    boxes = interpreter.get_output_details()
    boxes_es, boxes_zp = output_details[1]['quantization']
    boxes = interpreter.get_tensor(output_details[1]['index'])
    boxes = (boxes.astype(np.float32) - boxes_zp) * boxes_es
    boxes = boxes[0]
    
    # Classes
    classes = interpreter.get_output_details()
    classes_es, classes_zp = output_details[3]['quantization']
    classes = interpreter.get_tensor(output_details[3]['index'])
    classes = (classes.astype(np.float32) - classes_zp) * classes_es
    classes = classes[0]
    
    # Scores
    scores = interpreter.get_output_details()
    scores_es, scores_zp = output_details[0]['quantization']
    scores = interpreter.get_tensor(output_details[0]['index'])
    scores = (scores.astype(np.float32) - scores_zp) * scores_es
    scores = scores[0]
    
    for i in range(len(scores)):
        if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
    
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * alt)))
            xmin = int(max(1,(boxes[i][1] * larg)))
            ymax = int(min(alt,(boxes[i][2] * alt)))
            xmax = int(min(larg,(boxes[i][3] * larg)))
            
            if int(classes[i]) == 0:
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 255, 50), 1)
            else:
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255, 50, 50), 1)
    
            # Draw label
            object_name = labels[int(classes[i])] # Busca o nome do label atual
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Adiciona a % ao label atual
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Tamanho da fonte
            label_ymin = max(ymin, labelSize[1] + 10) # Offset p/ caso o texto esteja muito próximo da borda superior
            cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Caixa p/ Texto
            cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1) # Escreve o label



    # Draw framerate in corner of frame
    cv2.putText(img,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', img)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()