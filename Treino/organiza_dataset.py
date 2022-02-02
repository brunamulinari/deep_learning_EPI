import os
import shutil
import glob
import xml.etree.ElementTree as ET
import random
import io
from math import ceil
import cv2
import numpy as np
from PIL import Image

# Necessário compilar a API de Object Detection do Tensorflow
from object_detection.utils import dataset_util

# Importa V1 do TF (Segundo exemplo da documentação p/ conversão)
import tensorflow.compat.v1 as tf
###############################################################################

def separa_relevantes(pasta):
    ''' 
    Função que separa os arquivos com base nos labels
    '''
    
    # Cria pasta "Separados", caso não exista
    destino = os.path.join(pasta, 'Separados')
    if not os.path.exists(destino):
        os.makedirs(destino)
      
    # Realiza leitura dos arquivos de anotações    
    for xml in glob.glob(pasta + '/*.xml'):        
        dados = ET.parse(xml)
        dados = dados.getroot()
        arquivo = dados.find('filename').text
        '''
        Todas as imagens do dataset possuem extensão '.jpg', enquanto o arquivo de anotações aponta para um arquivo '.jpeg'
        Aqui é feita a correção para estes casos
        '''
        if arquivo.lower().endswith('.jpeg'):
            arquivo = arquivo[:-5] + '.jpg'

        # Verifica se a imagem possui pelo menos uma das anotações relevantes
        for objeto in dados.findall('object'):
            tipo_objeto = objeto.find('name').text
            if tipo_objeto in ['person', 'hat']:
                # Move a imagem e suas anotações p/ pasta destino
                nome_arq = xml.split('\\')[-1]
                shutil.move(xml, os.path.join(destino, nome_arq))               
                shutil.move(os.path.join(pasta, nome_arq[:-4] + '.jpg'), os.path.join(destino, nome_arq[:-4] + '.jpg'))
                break

    
    # Deleta sobras 
    sobras = [f for f in os.listdir(pasta) if f.endswith('.xml')]
    sobras_img = [f for f in os.listdir(pasta) if f.endswith('.jpg')]
    for sobra in sobras:
        os.remove(os.path.join(pasta, sobra))
    for sobra in sobras_img:
        os.remove(os.path.join(pasta, sobra))
             
   
def separa_grupos(pasta): 
    '''
    Função que separa os arquivos em treino e teste (80/20)
    '''   
    
    pasta_separados = os.path.join(pasta, 'Separados')
    pasta_treino = os.path.join(pasta_separados, 'treino')
    pasta_teste = os.path.join(pasta_separados, 'teste')
    
    # Cria pastas de treino e teste, caso não existam
    if not os.path.exists(pasta_treino):
        os.makedirs(pasta_treino)
    if not os.path.exists(pasta_teste):
        os.makedirs(pasta_teste)        
                
    arquivos = [f for f in os.listdir(pasta_separados) if f.endswith('.xml')]
    # Embaralha os exemplos 
    random.shuffle(arquivos)
    
    for contador, arquivo in enumerate(arquivos):    
        img = os.path.join(pasta_separados, arquivo[:-4] + '.jpg')
        lbl = os.path.join(pasta_separados, arquivo)
        
        # Verifica se o arquivo será utilizado p/ treino ou teste (80/20)
        if contador <= ceil(len(arquivos)*0.8):
            destino = pasta_treino
        else:
            destino = pasta_teste
        
        # Move a imagem e suas anotações
        shutil.move(img, destino)
        shutil.move(lbl, destino)


def constroi_tfrecord(pasta):
    '''
    Função p/ converter o dataset p/ o tipo TFRecord (p/ ser utilizado p/
    detecção de imagens)
    
    '''
    
    pasta_separados = os.path.join(pasta, 'Separados')
    for tipo in ['treino', 'teste']:       
        pasta = os.path.join(pasta_separados, tipo)
        
        # Inicia escritor de TFRecords
        writer = tf.python_io.TFRecordWriter(os.path.join(pasta_separados, tipo+'.record'))

        # Inicia listas das coordenadas das BBox
        xmins = []; xmaxs = []; ymins = []; ymaxs = []; 
       
        # Inicia a lista de classes, nomes dos arquivos e imagens
        tipos = []; nomes = []; imgsv = []
        for xml in glob.glob(pasta + '/*.xml'):
            # Realiza leitura do arquivo de anotações
            dados = ET.parse(xml)
            dados = dados.getroot()
            arquivo = xml[:-4] + '.jpg'
            
            # Busca cada objeto na imagem (person [pessoa sem capacete] & hat [pessoa com capacete])
            txmin = []; txmax = []; tymin = []; tymax = []; ttipo = []
            for objeto in dados.findall('object'):
                tipo_objeto = objeto.find('name').text
                if tipo_objeto in ['person', 'hat']:
                    bbox = objeto.find('bndbox')
                    ttipo.append(tipo_objeto)
                    txmin.append(int(bbox.find('xmin').text))
                    txmax.append(int(bbox.find('xmax').text))
                    tymin.append(int(bbox.find('ymin').text))
                    tymax.append(int(bbox.find('ymax').text))                   
            
            if len(ttipo) > 0:
                nomes.append(arquivo)
                xmins.append(np.array(txmin))
                xmaxs.append(np.array(txmax))
                ymins.append(np.array(tymin))
                ymaxs.append(np.array(tymax))
                ttipo = ['sem_capacete' if x=='person' else 'com_capacete' for x in ttipo]
                tipos.append(np.array(ttipo))
                
                # Abre imagem referente as anotações atuais
                with tf.gfile.GFile(xml[:-4] + '.jpg', 'rb') as img:
                    img = img.read()                     
        
                imgsv.append(img)          
                              
        # Escreve os records              
        for nome, xmin, xmax, ymin, ymax, tipo, img in zip(nomes,
                                                      xmins,
                                                      xmaxs,
                                                      ymins,
                                                      ymaxs,
                                                      tipos,
                                                      imgsv):
            
            img_io = io.BytesIO(img)
            img_pl = Image.open(img_io)
            largura, altura = img_pl.size
            labels = [1 if x == 'sem_capacete' else 2 for x in tipo]
            # Propriedades p/ montar o dicionário p/ TFRecord
            p_cl = 'image/object/class/'
            p_bb = 'image/object/bbox/'
            info_arq = {
                        p_cl + 'text': dataset_util.bytes_list_feature(np.char.encode(tipo, 'utf8')),
                        p_cl + 'label': dataset_util.int64_list_feature(labels),                  
                        
                        p_bb + 'xmin': dataset_util.float_list_feature(xmin/largura),
                        p_bb + 'xmax': dataset_util.float_list_feature(xmax/largura),
                        p_bb + 'ymin': dataset_util.float_list_feature(ymin/altura),
                        p_bb + 'ymax': dataset_util.float_list_feature(ymax/altura),
                    
                        'image/height': dataset_util.int64_feature(altura),
                        'image/width': dataset_util.int64_feature(largura),
                        'image/filename': dataset_util.bytes_feature(nome.encode('utf8')),
                        'image/source_id': dataset_util.bytes_feature(nome.encode('utf8')),
                        'image/encoded': dataset_util.bytes_feature(img),
                        'image/format': dataset_util.bytes_feature(b'.jpg'),                        
                        }
            
            
            tf_obj = tf.train.Example(features=tf.train.Features(feature=info_arq))
            writer.write(tf_obj.SerializeToString())

        # Fecha o escritor de TFRecords
        writer.close()


            

        
separa_relevantes("E:\\Estudos\\Proj_DL\\V320\\Dataset")
separa_grupos("E:\\Estudos\\Proj_DL\\V320\\Dataset")
constroi_tfrecord("E:\\Estudos\\Proj_DL\\V320\\Dataset")

