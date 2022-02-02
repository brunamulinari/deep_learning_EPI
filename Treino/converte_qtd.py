import tensorflow as tf
import os
import cv2

# P/ fazer a quantização, é necessário apresentar uma parcela significativa do dataset (Isto p/ determinação de valores máximos e mínimos dos pesos e etc convertidos p/ uint8

ds_rel = 200 # Quantidade de imagens a serem utilizadas na quantização (n+1)


caminho_imagens = 'E:\\Estudos\\Proj_DL\\V320\\Dataset\\Separados\\treino'
arquivos = [f for f in os.listdir(caminho_imagens) if (f.endswith('.jpg') or f.endswith('.png'))]
lista_imagens = []
for idx, img in enumerate(arquivos):
    temp_img = os.path.join(caminho_imagens, img)
    temp_img = (cv2.imread(temp_img) / 127.5) -1
    temp_img = cv2.resize(temp_img, (320, 320))
    lista_imagens.append(temp_img.astype('float32'))
    if idx > ds_rel:
        break

# Cria o dataset
dataset = tf.data.Dataset.from_tensor_slices((lista_imagens)).batch(1)

# Deleta a variável temporária com a lista de imagens (evitar estouro de memória)
del lista_imagens

# Cria o iterador do dataset
def iterador_dataset():
    for img in dataset.take(ds_rel):
        yield [img]

conversor = tf.lite.TFLiteConverter.from_saved_model('E:\Estudos\Proj_DL\V320\Modelo_Graph\saved_model')
conversor.optimizations = [tf.lite.Optimize.DEFAULT]
conversor.representative_dataset = iterador_dataset  
conversor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # Habilita opção de quantização
conversor.target_spec.supported_types = [tf.int8]

# Converte também as entradas e saídas do modelo p/ uint8 [para poder compilar para TPU]
conversor.inference_input_type = tf.uint8 
conversor.inference_output_type = tf.uint8
 
modelo_quantizado = conversor.convert()

tflite_model = conversor.convert()
with open('convertido_qtd.tflite', 'wb') as f:
  f.write(tflite_model)