# Carrega bibliotecas necessárias
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io


# Carrega o modelo e adquire informações das entradas
interpreter = tflite.Interpreter(model_path='modelo1.tflite')
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
img_cvt = np.expand_dims(img_cvt, axis=0) # Corrige p/ o tensor ([x, y, c] -> [1, x, y, c])


# Apresenta imagem p/ o modelo
interpreter.set_tensor(input_details[0]['index'], img_cvt)
interpreter.invoke()


# Extrai os resultados
output_details = interpreter.get_output_details()
boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
classes = interpreter.get_tensor(output_details[3]['index'])[0] 
scores = interpreter.get_tensor(output_details[0]['index'])[0]

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


        

img.save('resultado.png')