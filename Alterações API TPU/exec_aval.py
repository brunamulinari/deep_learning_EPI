import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import glob
import os
import platform
import argparse
import shutil
from datetime import datetime
from calculos import avalia_modelo

parser = argparse.ArgumentParser()
parser.add_argument('--modelo', type=str, help="Modelo a ser usado (1, 2 ou 3)")
args = parser.parse_args()

modelo_num = args.modelo

if modelo_num == None:
    modelo_num = 1


if modelo_num == '3':
    _EDGETPU_SHARED_LIB = {
      'Linux': 'libedgetpu.so.1',
      'Darwin': 'libedgetpu.1.dylib',
      'Windows': 'edgetpu.dll'
    }[platform.system()]
    
    def load_edgetpu_delegate(options=None):
      """Loads the Edge TPU delegate with the given options.
      Args:
        options (dict): Options that are passed to the Edge TPU delegate, via
          ``tf.lite.load_delegate``. The only option you should use is
          "device", which defines the Edge TPU to use. Supported values are the same
          as `device` in :func:`make_interpreter`.
      Returns:
        The Edge TPU delegate object.
      """
      return tflite.load_delegate(_EDGETPU_SHARED_LIB, options or {})    

    interpreter = tflite.Interpreter(model_path='modelo3.tflite', experimental_delegates=[load_edgetpu_delegate({'device': 'usb'})])

else:
    interpreter = tflite.Interpreter(model_path='modelo' + modelo_num + '.tflite')
    
    
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

cam_raiz = os.getcwd()
cam_teste = os.path.join(cam_raiz, 'grupo_teste')

cam_imgs = os.path.join(cam_teste, 'imagens')
cam_anotacoes = os.path.join(cam_raiz, 'modelo' + modelo_num)

cam_resultados = os.path.join(cam_raiz, 'resultados')


if not(os.path.exists(cam_anotacoes)):
    os.makedirs(cam_anotacoes)
if not(os.path.exists(cam_resultados)):
    os.makedirs(cam_resultados)

# Carrega dataset de teste

imagens = glob.glob(cam_imgs + '/*.jpg')

# Inicializa temporizador
tempo_ini = datetime.now()

qtd_max = 0

for img in imagens:
    nome = img.split('/')[-1].split('.')[0]
    img = cv2.imread(img)
    alt, larg, _ = img.shape
    img_cvt = (img.astype('float32')/127.5)-1 # Limites entre -1 e 1
    img_cvt = cv2.resize(img_cvt, (320, 320)) # Ajusta o tamanho p/ entrada do modelo (320x320)
    img_cvt = cv2.cvtColor(img_cvt, cv2.COLOR_BGR2RGB) # Ajusta p/ formato de entrada (RGB)
    
    # Aplica quantizacao caso esteja avaliando via modelo 2 ou 3
    if modelo_num == '2' or modelo_num == '3':
        es, zp = input_details[0]['quantization']
        img_cvt = img_cvt / es + zp
        img_cvt = img_cvt.astype(input_details[0]['dtype'])        
    
    
    img_cvt = np.expand_dims(img_cvt, axis=0) # Corrige p/ o tensor ([x, y, c] -> [1, x, y, c])
    
    # Apresenta imagem p/ o modelo
    interpreter.set_tensor(input_details[0]['index'], img_cvt)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    
    if modelo_num == '1':        
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
        classes = interpreter.get_tensor(output_details[3]['index'])[0] 
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
    else: 
        boxes = interpreter.get_output_details()
        boxes_es, boxes_zp = output_details[1]['quantization']
        boxes = interpreter.get_tensor(output_details[1]['index'])
        boxes = (boxes.astype(np.float32) - boxes_zp) * boxes_es
        boxes = boxes[0]        
        classes = interpreter.get_output_details()
        classes_es, classes_zp = output_details[3]['quantization']
        classes = interpreter.get_tensor(output_details[3]['index'])
        classes = (classes.astype(np.float32) - classes_zp) * classes_es
        classes = classes[0]
        scores = interpreter.get_output_details()
        scores_es, scores_zp = output_details[0]['quantization']
        scores = interpreter.get_tensor(output_details[0]['index'])
        scores = (scores.astype(np.float32) - scores_zp) * scores_es
        scores = scores[0]        


    
    with open(os.path.join(cam_anotacoes, nome + '.txt'), 'w') as f: 
        contador = 0
        for i in range(len(scores)):
            if ((scores[i] > 0.3) and (scores[i] <= 1.0)):
                contador+=1
                ymin = str(int(max(1,(boxes[i][0] * alt))))
                xmin = str(int(max(1,(boxes[i][1] * larg))))
                ymax = str(int(min(alt,(boxes[i][2] * alt))))
                xmax = str(int(min(larg,(boxes[i][3] * larg))))
                                
                if classes[i] == 0:
                    obj = 'sc'
                elif classes[i] == 1:
                    obj = 'cc'
                else:
                    continue
                
                f.write(obj + ' ' + str(scores[i]) + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + '\n')
    
        if contador>qtd_max:
            qtd_max = contador            


tempo_fin = datetime.now()

avalia_modelo(modelo_num)

with open(os.path.join(cam_resultados, 'modelo' + str(modelo_num) + '.txt'), 'a') as arq:
    arq.write('\n\n\nTempo total de inferencia: ' + str(tempo_fin - tempo_ini))
    arq.write('\n\n\nQuantidade max de detecoes: ' + str(qtd_max))
    

print('Tempo total de execucao: ', tempo_fin-tempo_ini)

shutil.rmtree(os.path.join(os.getcwd(), '.temp_files'))
shutil.rmtree(os.path.join(os.getcwd(), '__pycache__'))
