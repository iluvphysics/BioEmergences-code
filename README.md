# BioEmergences-code

#### Patches_emb :
Fichier servant à exploiter le fichier EMB en sortie de l'algorithme de tracking. Créations des patch avec les tâches gaussiennes, des données d'entrainement et de test pour la détéection de divisions cellulaires. Identification des cellules qui se divisent et leurs cellules filles pour chaque pas de temps.

#### vtk_to_tiff : 
Fonctions permettant de convertir des fichiers VTK, ceux utilisés en sortie du microscope, à des fichiers TIF compatibles avec CSBDeep. Génere les données
              d'entraînement pour la prédiction de pas de temps intermédiaires.

#### tracking
Pour la détection de divisions cellulaires.

entrainement tracking: Configuration et entrainement avec les fonctions de CSBDeep

evaluation tracking : fonctions permettant de faire prédire des images au modèle, et de l'évaluer sur les données de test.

#### denoising
Idem, pour la prédiction de pas de temps intermédiaire

#### tif2vtk.ijm
Macro Fiji pour convertir les fichier TIF en fichiers VTK, et modifier le contraste.
