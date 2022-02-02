import tensorflow as tf

conversor = tf.lite.TFLiteConverter.from_saved_model('E:\Estudos\Proj_DL\V320\Modelo_Graph\saved_model')
modelo = conversor.convert()

tflite_model = conversor.convert()
with open('convertido.tflite', 'wb') as f:
  f.write(modelo)