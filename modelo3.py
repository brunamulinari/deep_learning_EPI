# Carrega bibliotecas necessárias
import tflite_runtime.interpreter as tflite
import plataform
from PIL import Image, ImageDraw, ImageFont
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
interpreter = tflite.Interpreter(model_path='modelo3.tflite', experimental_delegates=load_edgetpu_delegate({'device': 'usb'}))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

# Inicializa nomes dos labels
labels = ['sc', 'cc']
  
# Carrega uma imagem para teste e adequa
img = Image.open('1.jpg')
larg, alt = img.size
img2 = img
img2 = img2.resize((320,320))
img_cvt = np.array(img2.getdata()).reshape(img2.size[0], img2.size[1], 3)
img_cvt = (img_cvt.astype('float32')/127.5)-1 # Limites entre -1 e 1

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


draw = ImageDraw.Draw(img)
for i in range(len(scores)):

    if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
        # Busca coordenadas do objeto atual. É possível que as coordenadas retornadas estejam fora
        # dos limites da imagem, por isso é forçado com "max e min"
        ymin = int(max(1,(boxes[i][0] * alt)))
        xmin = int(max(1,(boxes[i][1] * larg)))
        ymax = int(min(alt,(boxes[i][2] * alt)))
        xmax = int(min(larg,(boxes[i][3] * larg)))

        if classes[i] == 0:
            cor = 'red'
        else:
            cor = 'blue'

        draw.rectangle([(xmin, ymin), (xmax, ymax)],
            outline=cor)
        draw.text((xmin + 10, ymin + 10),
            labels[int(classes[i])] + ' ' + str(round(scores[i]*100, 2)) + '%',
            font = ImageFont.truetype('arial.ttf', 16), fill=cor)


img.save('resultado_qtd.png')
        
        
