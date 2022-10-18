import tensorflow
from tensorflow import keras as nn
import numpy as np
import h5py
from skimage import io

"""
mLoad_dataset:
    Funcion que nos ayuda a crear el tensor de imagenes, de canal 1, dada una lista de 
    paths del GT de imagenes. Aqui se reestructura cada array, tanto al insertar un pad
    constante en imagenes con menor shape, y con una dimension adicional que se usa como
    el cana de imagen. El objetivo final es no tener problemas durante la cocatencacion
    de todos estos tensores.  

"""
def mLoad_GT(paths):
    
    #Cargamos el array usando su ubicacion. Sabemos que esta guardados en un h5file
    GT_lts = []
    for path in paths:
        f = h5py.File(path, 'r')
        GT_lts.append(np.asarray([f['density']]))
        f.close
    
    #Obtenemos el shape maximo 
    mx, my = max_shape(GT_lts)
    
    
    for i in range(len(GT_lts)):
        #Anadimos el path
        GT = GT_lts[i]
        GT = tensorflow.pad(GT[0,:,:], pad_Tensor([mx, my], GT.shape[1:]), "CONSTANT")
        GT = np.asarray([GT])
        
        #Anadimosla dimension extra
        swapped = np.moveaxis(GT, 0, 2)  
        arr4d = np.expand_dims(swapped, 0)
        GT_lts[i] = arr4d    
        
    
    tt = nn.layers.Concatenate(axis=0)(GT_lts)  
    return tt
    
def mLoad_Img(paths):
    
    #Cargamos el array usando su ubicacion. Sabemos que esta guardados en un h5file
    GT_lts = []
    for path in paths:
        img_path = str(path).replace('.h5','.jpg').replace('ground_truth_density','images').replace('GT_IMG_', 'IMG_')
        img = io.imread(img_path, as_gray=True)
        GT_lts.append(np.asarray([img.astype(np.float64)]))
    
    #Obtenemos el shape maximo 
    mx, my = max_shape(GT_lts)
    
    
    for i in range(len(GT_lts)):
        #Anadimos el path
        GT = GT_lts[i]
        GT = tensorflow.pad(GT[0,:,:], pad_Tensor([mx, my], GT.shape[1:]), "CONSTANT")
        GT = np.asarray([GT])
        
        #Anadimosla dimension extra
        swapped = np.moveaxis(GT, 0, 2)  
        arr4d = np.expand_dims(swapped, 0)
        GT_lts[i] = arr4d    
        
    
    tt = nn.layers.Concatenate(axis=0)(GT_lts)  
    return tt
    
"""
max_shape:
    Funcion que nos ayuda a encontrar el shape maximo dada una lista de imagenes.
    Para esto, se crea una lista con la su longitud en cada una de las dimensiones y
    se retorna el maximo alor para cada una de estas listas.
"""
def max_shape(shape_lts):
    shp_x = []
    shp_y = []

    for ff in shape_lts:
        shp_x.append(ff.shape[1])
        shp_y.append(ff.shape[2])
    
    return max(shp_x), max(shp_y)
        

"""
pad_Tensor:
    Funcion que nos ayuda a ajustar el shape de cada imagen en base al shape maximo
    de todas las imagenes. El ajuste se realiza usando un pad constante y que centra 
    la imagen. El objetivo es evitar problemas al concatenar todas las imagenes en 
    un solo tensor.

"""
def pad_Tensor(max_shp, mshape):
    mx, my = max_shp
    
    pd_x = abs(mshape[0] - mx)
    pd_y = abs(mshape[1] - my)
    
    PD_X = []
    PD_Y = []
    
    if (np.mod(pd_x, 2)) == 1:
        pd_x = int(pd_x/2)
        PD_X = [pd_x + 1, pd_x]
    else:
        pd_x = int(pd_x/2)
        PD_X = [pd_x, pd_x]
    
    if (np.mod(pd_y, 2)) == 1:
        pd_y = int(pd_y/2)
        PD_Y = [pd_y + 1, pd_y]
    else:
        pd_y = int(pd_y/2)
        PD_Y = [pd_y, pd_y]
    
    return tensorflow.constant([PD_X, PD_Y])
        
        
        
    