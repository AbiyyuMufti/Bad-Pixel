""" 
/*
 * @Author: Julian Schweizerhof
 * @Email: diegruppetg@gmail.com
 * @Date: 2020-05-26 20:13:55
 * @Last Modified by: JLS666
 * @Last Modified time: 2020-05-26 20:35:38
 * @Description: Python Programm um ein oder mehrere Bilder des Formats his zu importieren, To-Do: Datei in OpenCV Format importieren
 */
 """

import numpy as np                                                          # Für Arrays
import cv2                                                                  # Import OpenCV
import os                                                                   # Für die Path-Manipulation
#import matplotlib.pyplot as plt

#how to import numpy und ov2.  Python updaten, pip install, pip3 install numpy, pip3 install opencv-python.

import detection
import config

detection.test(3)
detection.test(config.Bildhoehe)


def getNumberImages(pImportPath, rows, cols):                               # Funktion: Die Anzahl der Bilder in der Datei bestimmten, Rückgabewert: Anzahl Bilder
    file = open(pImportPath,'rb')                                           # File erneut öffnen, da ansonsten der "Cursor" falsch liegt
    data = np.fromfile(file,dtype=np.uint16)                                # komplettes File einlesen
    numberImages = (np.size(data) - 50) / (rows*cols)                       # Die Anzahl der Bilder bestimmen: die Größe des uint16 Arrays bestimmen, dann minus 50 Elemten (der Header der Datei) und dann geteilt durch die Pixel-Anzahl eines Bildes
    file.close()                                                            # File schließen
    return numberImages                                                     # Die Anzahl der Bilder zurückgeben

def hisImportFunction(pImportPath, pExport = False):                        # Funktion: Bilder im HIS-Format importieren, Übergabewert: Path zum Bild
    pathWithoutExtension = os.path.splitext(pImportPath) [0]                # Pfad ohne Dateiendung erzeugen, .his wird entfernt
    print("\n\n*************************************************************")
    print("Funktion zum Einlesen von HIS-Dateien aufgerufen")
    print("*************************************************************\n")
    fileName, fileExtension = os.path.splitext(os.path.basename(importPath))# Dateinamen und Dateiendung extrahieren    
    print("Die Datei", fileName, "wird jetzt eingelesen.")
    fd = open(pImportPath,'rb')                                             # Das Bild öffnen im "rb"-Modus: read binary
    data = np.fromfile(fd,dtype=np.uint16, count=50)                        # Den Header 50 mal mit unsinged int 16 Bit einlesen (erste 100 Bytes)
    rows = int(np.take(data, 8))                                            # Reihen bestimmen, in int konvertieren, ansonsten overflow Error bei der Funktion fromfile()
    cols = int(np.take(data, 9))                                            # Spalten bestimmen
    numberImages = int(getNumberImages(pImportPath, rows, cols))            # Anzahl der Bilder in der Datei bestimmen
    print("Ihre Datei hat", rows, "Reihen und", cols, "Spalten und besteht aus", numberImages, "Bild(ern)")



    for index in range(numberImages):                                       # Alle Bilder anzeigen und speichern
        f = np.fromfile(fd, dtype=np.uint16, count=rows*cols)               # Pixel lesen und in einem ein dimensionales Array speichern
        im = f.reshape((rows, cols)) 
                                              # Array in zwei dimensionales Array mit rows x cols erstellen
        #np.savetxt("foo.csv", im, delimiter=";")

        #for testValue in np.nditer(f):
        #    if testValue >= 60000:
        #        print("Sehr schwarz hier!")
        #plt.plot(im)
        #plt.show()
        if pExport == True:
            cv2.imshow('image', im)                                             # Array als Bild anzeigen
            cv2.imwrite(pathWithoutExtension+"_"+str(index)+'_beta.png',im, [cv2.IMWRITE_PNG_COMPRESSION,0])     # Array als PNG speichern ohne Kompression
            print("Ihre Datei wurden unter", pathWithoutExtension+"_"+str(index)+".png gespeichert")
            cv2.waitKey()                                                       # Warten bis eine Taste gedrückt wird      
    if pExport == True:
        cv2.destroyAllWindows()                                                 # Alle Fenster schließen    
    fd.close()                                                              # File schließen
    return im

    

#importPath = '/Users/julian/Desktop/test/Bildserie1_160kV_0uA.his' 
#importPath = "/Users/julian/Desktop/test/Bildserie1_160kV_0uA.his"
#importPath = "/Users/julian/Desktop/test/Bildserie2_160kV_70uA.his"

#importPath = "/Users/julian/Desktop/Bildserie1_160kV_0uA.his"
#importPath = "Bildserie1_160kV_0uA.his"
#bildArray = hisImportFunction(importPath,True)
#print(bildArray)

#importPath = "/Users/julian/Google Drive/Studium/Master/1. Semester/Mechatronische Systeme/Mecha. Systeme/F&E Bad-Pixel/2. Stand der Technik - Recherche/Beispielbilder/Daten/Aufnahmen zur Korrektur Panel Version 2/Serie4/Bildserie4_75kV_20uA.his"
importPath = "Bildserie3_160kV_0uA.his"


#ab hier Quasi die main:
detection.DeadPixelFinder(hisImportFunction(importPath))
