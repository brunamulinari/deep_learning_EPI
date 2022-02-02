# Carrega bibliotecas necessárias
import tflite_runtime.interpreter as tflite
import plataform
import cv2
import numpy as np
import io


# Carregamento p/ TPU


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




# Carrega o modelo e adquire informações das entradas
# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='modelo3.tflite', experimental_delegates=[load_edgetpu_delegate({'device': 'usb'})])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

# Inicializa nomes dos labels
labels = ['sc', 'cc']
  
# Carrega uma imagem para teste e pré-processa
img = cv2.imread('1.jpg')
alt, larg, _ = img.shape
img_cvt = (img.astype('float32')/127.5)-1 # Limites entre -1 e 1
img_cvt = cv2.resize(img_cvt, (320, 320)) # Ajusta o tamanho p/ entrada do modelo (320x320)
img_cvt = cv2.cvtColor(img_cvt, cv2.COLOR_BGR2RGB) # Ajusta p/ formato de entrada (RGB)
img_cvt = np.expand_dims(img_cvt, axis=0) # Corrige p/ o tensor ([x, y, c] -> [1, x, y, c])

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
        
        
cv2.imwrite('resultado_qtd_tpu.png', img)