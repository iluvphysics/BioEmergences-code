#!/usr/bin/env python
# coding: utf-8

# In[3]:


from PIL import Image
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy
import vtk
import gzip
from os.path import exists
import shutil
import os

def liste_temps(date: int, dirpath: str, ch: int, formats=".tif") :
    """ trouve la liste des pas de temps pour chaque fichier dans dirpath """

    fichiers_liste = os.listdir(dirpath)
    return_temps = []
    
    # on ne sélectionne que les fichiers d'image
    for file in fichiers_liste :
        if file[:len(date)+2] != date+"_t" or file[-2-len(str(ch).zfill(2))-len(formats):] != "ch"+str(ch).zfill(2)+formats :
            fichiers_liste.remove(file)
            print(file, "Mauvais fichier")
    print(len(fichiers_liste))      
    if fichiers_liste[-1][-2-len(str(ch).zfill(2))-len(formats):] != "ch"+str(ch).zfill(2)+formats :
        print(fichiers_liste[-1], "Mauvais fichier")
        fichiers_liste = fichiers_liste[:-1]
    
    for file in fichiers_liste :
        
        # compteur 
        s= 0
                    
        # on augmente la longueur du segment jusqu'à tomber sur _
        while file[len(date) + 2 + s] != "_" :
            s += 1
        
        return_temps.append( int(file[len(date)+2: len(date) + 2 + s]) )
    
    return np.array(return_temps)

def gunzip(file_path,output_path):
    with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def vtkToArray(nomVtk) :
    
    file_name = nomVtk
    
    # si le fichier est compressé, on le décompresse
    if nomVtk[-3:] == ".gz" :
        gunzip(nomVtk, nomVtk[:-3])
        file_name = nomVtk[:-3]
        
    #extraction des données
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_name)
    reader.Update()
    data = reader.GetOutput()
    
    #dimensions des données
    dims = data.GetDimensions()
    nx, ny, nz = dims
    #print("dims, nx, ny, nz: ", dims)
    
     #métadonnées 
    spacing_array = np.float32(data.GetSpacing())
    #print("spacing: ",spacing_array)
    
    # Conversion en tableau Numpy
    data_vtk = data.GetPointData().GetArray(0)
    data_numpy = np.array([data_vtk.GetTuple(i) for i in range(nx * ny * nz)]).astype('uint16')
    data_numpy = data_numpy.reshape((nz, ny, nx))
    
    if exists(nomVtk[:-3]) :
        os.remove(nomVtk[:-3])

    return data_numpy, tuple(spacing_array)

def save_imagej_tiff(save_path: str, data: np.ndarray, scale: tuple[float, ...], units: str, com=None):
    """Save image as tiff to path or buffer
    :param scale: image scale
    :param data: image for save
    :param units: units of image
    :param save_path: save location
    """
    
    if exists(save_path) :
        os.remove(save_path)
        
    assert data.dtype in [np.uint8, np.uint16, np.float32]
    metadata: dict[str, typing.Any] = {"mode": "color", "unit": units, 'axes': 'ZYX' }
    if len(scale) >= 3:
        metadata["spacing"] = scale[-1]
    resolution = [1 / x for x in scale[:-1]]
    tif.imwrite(
        save_path,
        data,
        imagej=True,
        metadata=metadata,
        resolution=resolution,
    ) 
    
def save_imagej_tiff4d(save_path: str, data: np.ndarray, scale: tuple[float, ...], units: str, com=None):
    if exists(save_path) :
        os.remove(save_path)
        
    assert data.dtype in [np.uint8, np.uint16, np.float32]
    metadata: dict[str, typing.Any] = {"mode": "color", "unit": units, 'axes': 'ZCYX'}
    if len(scale) >= 3:
        metadata["spacing"] = scale[-1]
    resolution = [1 / x for x in scale[:-1]]
    tif.imwrite(
        save_path,
        data,
        imagej=True,
        metadata=metadata,
        resolution=resolution,
    )  # , compress=6,
    
#save_imagej_tiff("test2.tif", tableau, scale= spacing, units="nm", com=None)
def vtkToTiff(nomVtk: str, nomTif: str):
    
    if exists(nomTif):
        os.remove(nomTif)

    tableau, spacing = vtkToArray(nomVtk)
    
    # sauvegarde de l'image à partir du tableau 3d créé
    save_imagej_tiff(nomTif, tableau, spacing, units="um", com=None)

    return None

def vtkTiff_dir(date, vtk_dir, output_dir, tstart, tstop, tstep, demistep=False) :
    """ Convertit les pas de temps {tstart,..., tstop-1} de vtk.gz en tif. 
    demistep: divise les pas de temps sauvegardés par 2 lors du nommage du fichier
    date: nom du dataset utilisé """
    
    indt = -1
    files = os.listdir(vtk_dir)
    vtk = os.listdir(vtk_dir)
    
    # on élimine les éléments n'étant pas des vtk.gz
    for im in vtk :
        if im[-7:] != '.vtk.gz' or im[:len(date)] != date :
            files.remove(im)
    vtk = files
    
    # on cherche l'indice de vtk correspondant au pas de temps tstart
    for i, im in enumerate(vtk) :
        s = 0
        while im[len(date)+2+s] != '_' :
            s+= 1
        if int(im[len(date)+2: len(date)+2+s]) == tstart :
            indt = i
    assert indt != -1, "Pas de temps t={} pas trouvé.".format(tstart)
    
    # pour chaque pas de temps, on convertit en tif et on sauvegarde dans output_tif
    for t in range(tstart, tstop, tstep) :
        nomTif = vtk[indt][:-7] + ".tif"
        
        if demistep :
            s = 0
            while nomTif[len(date)+2+s] != "_" :
                s += 1
            num_t = str(int(nomTif[len(date)+2:len(date)+2+s])//2).zfill(3)
            nomTif = vtk[indt][:len(date)+2] + num_t + vtk[indt][len(date)+2+len(num_t):-7] + ".tif"
        
        print(nomTif)
        vtkToTiff(vtk_dir+"/"+vtk[indt], output_dir+"/"+nomTif) 
        indt += tstep
            
        
        print('t =',t, '/', tstop-tstep)
        
    return None


# In[4]:


#inp_vtk = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_all/070418a"
#out_tif = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_all/tif_fullstep"
#
#vtkTiff_dir("070418a",inp_vtk, out_tif, tstart = 602, tstop = 722, tstep=2, demistep=True)


# In[5]:


#out = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_all/vtk_fullstep"

import shutil

#files = os.listdir(inp_vtk)
#vtk = os.listdir(inp_vtk)
#    
## on élimine les éléments n'étant pas des vtk.gz
#for im in vtk :
#    if im[-7:] != '.vtk.gz' or im[:len(date)] != date :
#        files.remove(im)
#vtk = files
#
#for im in vtk :
#    s = 0
#    while im[9+s] != "_" :
#        s += 1
#    num_t = int(im[9:9+s])
#    if num_t % 2 == 0 :
#        nvnom = im[:9] + str(num_t//2).zfill(3) + im[12:]
#        shutil.copy(inp_vtk+'/'+im, out+"/"+nvnom)
#shutil.copyfile(src, dst)


# In[ ]:


def tableauZCYX(array1, array2) :
    """ Transforme deux tableaux 3D en un tableau 4D """
    tableau_4d = np.zeros(shape= array1[:,0,0].shape + tuple([2]) + array1[0,:,:].shape )
    tableau_4d[:,0,:,:], tableau_4d[:,1,:,:] = array1, array2
    return tableau_4d.astype(np.uint16)

def SourcesTargetTiff(nbch_input, date, input_dir, output_dir, tstart, tstop, chout, demi_pas=False) :
    """ Transforme les fichiers de input_dir (VTK) en fichiers TIF dans output_dir """
        
    # dossier des fichiers vtk
    files_list = os.listdir(input_dir)
    
    ###initialisation
    
    t1, t2, t3 = str(tstart).zfill(3), str(tstart+1).zfill(3), str(tstart+2).zfill(3)
    #t_init = np.zeros(nbch_input+1)
    #for i in range(len(t_init)) :
    #    t_init[i] = str(tstart+i).zfill(3)
    
    # on cherche l'indice de la liste correspondant à t = tstart
    tindex = 0
    while files_list[tindex] != files_list[tindex][:len(date)+2] + t1 + files_list[tindex][len(date)+len(t1)+2:] :
        tindex += 1
    assert ( files_list[tindex] == files_list[tindex][:len(date)+2] + t1 + files_list[tindex][len(date)+2+len(t1):] )
    
    #f_init = []
    #for t in range(tindex, tindex+nbch_input+1) :
    #    f_init[t] = files_list[t]
    f1,f2,f3 = files_list[tindex], files_list[tindex+1], files_list[tindex+2]
    
    chin = f1[-9:-7]
    
    array1, sp = vtkToArray(input_dir+"/"+f1) # spacing utilisé: sp
    array2, _ = vtkToArray(input_dir+"/"+f2)
    array3, _ = vtkToArray(input_dir+"/"+f3)
    
    #array_init = np.empty(tuple([nbch_input + 1]) + array1.shape)
    #array_init[0] = array1
    #
    #for i in range(1, nbch_input) :
    #    array_init[i], _ = vtoToArray( input_dir+"/"+f_init[i] )
    
    
    # entrainement du réseau pour prédire un pas connu entre deux images données
    if not demi_pas :
        tableau_4d = tableauZCYX(array1, array3).astype(np.uint16)
        array4d = tableauZCYX(array2, array2).astype(np.uint16)
        
        # save
        save_imagej_tiff4d(output_dir+"/"+date+"_t"+t1+t3+"_ch"+str(chout).zfill(2)+"4D.tif", 
                           tableau_4d, scale=sp, units ="um", com=None )
        save_imagej_tiff4d(output_dir+"/"+date+"_t"+t2+"_ch"+str(chout).zfill(2)+"4D.tif",
                           array4d, scale=sp, units ="um", com=None )
        
    # entrainement du réseau pour prédire un pas intermédiaire
    else :
        tableau_4d = tableauZCYX(array1, array2)
        
        #tableau à 2 canaux des 2 premières images
        save_imagej_tiff4d(output_dir+"/"+date+"_t"+str(2*int(t1)+1).zfill(3)+"_ch"+str(chout).zfill(2)+".tif", 
                           tableau_4d, scale=sp, units ="um", com=None)
        #3 premiers tableaux
        save_imagej_tiff4d(output_dir+"/"+date+"_t"+str(2*int(t2)).zfill(3)+"_ch"+str(chout).zfill(2)+".tif", 
                           tableauZCYX(array2, array2), scale=sp, units ="um", com=None)
        save_imagej_tiff4d(output_dir+"/"+date+"_t"+str(2*int(t1)).zfill(3)+"_ch"+str(chout).zfill(2)+".tif", 
                           tableauZCYX(array1,array1), scale=sp, units ="um", com=None)
        save_imagej_tiff4d(output_dir+"/"+date+"_t"+str(2*int(t3)).zfill(3)+"_ch"+str(chout).zfill(2)+".tif", 
                           tableauZCYX(array3,array3), scale=sp, units ="um", com=None)
                
    # tableau auquel on va faire subir des permutations circulaires
    tableaux = np.zeros(shape= tuple([3]) + array1.shape)
    tableaux[0] = array1
    tableaux[1] = array2
    tableaux[2] = array3
    tableaux = tableaux.astype(np.uint16)
    
    print("tstart = "+str(tindex)+". Sauvegarde :", output_dir)
    
    for t in range(tstart+1, tstop-1):
       
        print( "t =",t, "/", tstop )
    
        # permutations
        tableaux[0] = tableaux[1]
        tableaux[1] = tableaux[2]
    
        # on a juste à lire le dernier fichier, en t+2
        # fichier en t+2
        f3 = input_dir+"/"+date + "_t" + str(t+2).zfill(3)+"_ch"+chin+".vtk.gz"
        
        tableaux[2], _ = vtkToArray(f3)
    
        # données à sauvegarder 
        if not demi_pas :
            tableau4d = tableauZCYX(tableaux[0], tableaux[2]).astype(np.uint16)
            tableau4dlabel = tableauZCYX(tableaux[1], tableaux[1]).astype(np.uint16)
            
            # noms des fichiers sauvegardés
            path_save4d = output_dir+"/"+date+"_t"+str(t).zfill(3)+str(t+2).zfill(3)+"_ch"+str(chout).zfill(2)+"4D.tif"
            path_savelabel = output_dir+"/"+date+"_t"+str(t+1).zfill(3)+"_ch"+str(chout).zfill(2)+"4D.tif"
            
            #save
            save_imagej_tiff4d(path_save4d, tableau4d, scale = sp, units="um", com=None)
            save_imagej_tiff4d(path_savelabel, tableau4dlabel, scale = sp, units="um", com=None)
            
        else :
            # tableaux 4d à utiliser pour les prédictions 
            tableau4d = tableauZCYX(tableaux[0], tableaux[1]).astype(np.uint16)
            
            save_imagej_tiff4d(output_dir+"/"+date+"_t"+str(2*t+1).zfill(3)+"_ch"+str(chout).zfill(2)+".tif",
                               tableau4d, scale=sp, units="um", com=None)
            save_imagej_tiff4d(output_dir+'/'+date+"_t"+str(2*(t+2)).zfill(3)+"_ch"+str(chout).zfill(2)+".tif",
                           tableauZCYX(tableaux[2], tableaux[2]), scale=sp, units='um', com=None)
    if demi_pas :
        # dernier tableau (4D) à sauvegarder 
        save_imagej_tiff4d(output_dir+"/"+date+"_t"+str(2*tstop - 1).zfill(3)+"_ch"+str(chout).zfill(2)+".tif",
                            tableauZCYX(tableaux[1], tableaux[2]).astype(np.uint16), scale=sp, units="um", com=None)
            
                  
    return None

def range_fichiers(date, dir_input ,dir_train) :
    """ renomme les fichiers et les ranges dans les jeux d'entrainement et de test """

    for filename in os.listdir(dir_input) :
        
        if len(filename) < 21 or filename[:len(date)+2] != date+"_t" :
            continue
    
        newname = filename
    
        if filename[-6:] == "4D.tif" :
            newname = filename[:-6] + ".tif"
        
        # données d'entrée du jeu d'entrainement
        if len(filename) == ( len(date) + len("_t") + 3 + len("_ch00.tif") ) + 3 :
            
            nbre = ( int(filename[len(date) + 2 : len(date) + 5]) + int(filename[len(date)+5: len(date)+8]) ) // 2 
            newname = newname[:len(date)+2] + str(nbre).zfill(3) + newname[len(date)+8:]
                        
            if os.path.exists(dir_train + "/train_samples/" + newname) :
                os.remove(dir_train + "/train_samples/" + newname)
                
            shutil.move(dir_input+"/"+filename, dir_train+"/train_samples/"+filename)
            os.rename(dir_train+"/train_samples/"+filename, dir_train+"/train_samples/"+newname)
        
        # étiquettes du jeu d'entrainement
        else :
            if os.path.exists(dir_train + "/train_labels/" + newname) :
                os.remove(dir_train + "/train_labels/" + newname)
                
            shutil.move(dir_input+"/"+filename, dir_train+"/train_labels/"+filename)
            os.rename(dir_train+"/train_samples/"+filename, dir_train+"/train_labels/"+newname)
            
    return None

def move_files(input_dir, output_dir, date, t:int, all_files=False) :
    """ déplace le(s) fichier(s) correspondant au pas de temps t, ou bien tout un répertoire """
    
    files = os.listdir(input_dir)
    
    if not all_files :
        for filename in files :
            
            if len(filename) < len(date) + 12 or filename[len(date)+2:] != date+"_t" :
                print("skip",filename)
                continue
            
            #compteur
            s = 0
            while filename[len(date)+2+s] != "_" :
                s += 1
                
            # on vérifie que le pas choisi correspond au fichier voulu
            if str(t).zfill(3) == filename[len(date) + 2: len(date)+2+s] :
                shutil.move(input_dir+"/"+filename, output_dir+"/"+filename)
    else :
        for filename in files :
            if len(filename) < len(date) + 12 or filename[len(date)+2:] != date+"_t" :
                print("skip",filename)
                continue
            shutil.move(input_dir+"/"+filename, output_dir+"/"+filename)
            
    return None


# In[ ]:


#date = "070418a"
#output_noyaux = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_noyaux"
#output_membranes = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_membranes"
#
#input_noyaux = "C:/Users/bioAMD/Desktop/Nathan/VTKdata070418/stock_noyaux"
#input_membranes = "C:/Users/bioAMD/Desktop/Nathan/VTKdata070418/stock_membranes"
##output_dir = "C:/Users/bioAMD/Desktop/Nathan/TIFFdata070418/stock_mix/tif_membranes"
#
#inp = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_all/070418a"
#
##SourcesTargetTiff(2, date, input_noyaux, output_noyaux,
##                 tstart=1, tstop=500, chout=8, demi_pas=True)
#SourcesTargetTiff(2, date, input_noyaux, output_noyaux, 
#                tstart=, tstop=500, chout=8, demi_pas=False)


# In[ ]:


# dir1 = "C:/Users/bioAMD/Desktop/Nathan/TIFFdata070418/stock_Demistep"
# dir2 = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataDemistep/inputDemistep"
# for t in range(2, 241) :
#     if t%2 != 0 :
#         move_files(dir1, dir2, '070418a',t, all_files=False)

# date = "070418a"
# dir_train_Uniform= "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataUniform/train"


# In[ ]:


# m = "C:/Users/bioAMD/Desktop/Nathan/TIFFdata070418/stock_Demistep/noyaux"
# for files in os.listdir(m) :
#     if files[:len(date)] == date and files[-4:] == '.tif':
#         nvnom = files[:-6] + "05" + files[-4:]
#         print(nvnom)
#     os.rename(m+"/"+files, m+"/"+nvnom)


##########  rangement des jeux d'entrainement et de test ##############
#date = "070418a"
#dir_membranes = "C:/Users/bioAMD/Desktop/Nathan/TIFFdata070418/stock_mix/tif_membranes"
#dir_noyaux = "C:/Users/bioAMD/Desktop/Nathan/TIFFdata070418/stock_mix/tif_noyaux"
#base_output = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix"
#
#files_noyaux = sorted(os.listdir(dir_noyaux))
#files_membranes = sorted(os.listdir(dir_membranes))
#
#scale = tuple(np.float32([1.34,1.34,1.34]))
#for n, m in zip(files_noyaux, files_membranes) :
#    
#
#    if n[len(date)+2:len(date)+2+snoy] != m[len(date)+2:len(date)+2+smem] :
#        continue 
#    print(n, "/", m)
#   
#   # pour chaque membrane/noyau, on les associe en un vecteur de 4 canaux 
#    xnoy, xmem = tif.imread(dir_noyaux+"/"+n), tif.imread(dir_membranes+"/"+m)
#    X = np.zeros(shape= xnoy[:,0,0,0].shape +tuple([4]) + xnoy[0,0,:,:].shape)
#
#   # 2 premiers canaux : noyaux. Deux derniers : membranes
#    X[:,0:2,:,:] = xnoy
#    X[:,2:4,:,:] = xmem
#    
#   # label : on garde uniquement l'image 3D du noyau 
#    if len(n) == 23 :
#        x = xnoy[:,0,:,:].astype(np.uint16)
#        newname = n[:-8] + "08.tif"
#        save_path = base_output + "/train/train_labels/" + newname
#        save_imagej_tiff(save_path, x, scale, units="um")
#       
#        print("t = "+newname[len(date)+2:len(date)+5]+" sauvegardé (label)")
#       
#   # sample : on garde tout et on renomme le fichier
#    elif len(n) == 26 :
#       
#       # nom du fichier à sauvegarder
#        nbre = ( int(n[len(date)+2:len(date)+5]) + int(n[len(date)+5:len(date)+8]) ) // 2
#        newname = n[:len(date)+2] + str(nbre).zfill(3) + n[len(date)+8:-8] + "08.tif"
#        save_path = base_output + "/train/train_samples/" + newname
#       
#        x = X.astype(np.uint16)
#        save_imagej_tiff4d(save_path, x, scale, units="um")
#        print("t = "+newname[len(date)+2:len(date)+5]+" sauvegardé (sample)")
#       
#    else :
#        print('erreur: mauvaise longueur')
#        break
###########################        
        


# In[ ]:


# patch_size = 32*64*64*2
# n_patches = 210*199
# print(2*patch_size*n_patches*4 * 1e-6)
# print((100/85)*100)


# In[ ]:


# import os
# import tifffile as tif 
# dir_samples = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/train/train_samples"
# dict_shape = dict()

# for file in os.listdir(dir_samples) :
    
#     x = tif.imread(dir_samples+"/"+file)
#     if x.shape not in dict_shape :
#         dict_shape[x.shape] = 1
#     else :
#         dict_shape[x.shape] += 1


# In[ ]:


# tstart = 543
# tstop = 548

# date= "070418a"
# scaling = 
# dir_input = "C:/Users/bioAMD/Desktop/Nathan/VTKdata070418/stock_noyaux"
# dir_output = "C:/Users/bioAMD/Desktop/Nathan/TIFFdata070418/frames_perdues"

# # SourcesTargetTiff(2, date, dir_input, dir_output, tstart, tstop,
# #                   chout=5, demi_pas=True)

    


# In[ ]:


# date = "070418a"
# dir_output = "C:/Users/bioAMD/Desktop/Nathan/TIFFdata070418/frames_perdues"


# In[ ]:


# path_noyaux = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_noyaux"
# path_membranes = "C:/Users/bioAMD/Desktop/Nathan/Interpolation/dataMix/dataDemistep/demistep_membranes"

# fn = os.listdir(path_noyaux)
# fm = os.listdir(path_membranes)

# print(fn)


# In[ ]:


#A = np.random.randint(low=0, high=10, size=(30,20,20))
#print(np.any(A == 9))


# In[ ]:


#plt.imshow(A[15,:,:], cmap='inferno')

