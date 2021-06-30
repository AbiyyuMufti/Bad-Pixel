"""
/*
 * @Author: Julian Schweizerhof
 * @Email: diegruppetg@gmail.com
 * @Date: 2020-06-15 17:48:37
 * @Last Modified by: JLS666
 * @Last Modified time: 2020-10-14 23:25:37
 * @Description: Python Programm um ein oder mehrere Bilder des Formats his zu importieren
 */
 """

#########################################################################################################
# HIS-File Aufbau:  100 Byte großer Header, danach unkomprimiertes Bild im Rohformat, z.B. in uint16
#                   Der 9. uint16 im Header: Anzahl der Pixel in der Breite
#                   Der 10. uint16 im Header: Anzahl der Pixel in der Höhe
#                   Der 11. uint16 im Header: Anzahl der Bilder in der Bildserie
#                   Über die Dateigröße kann die Farbtiefe berechnet werden          
#########################################################################################################

import os                                                                   # Für die Path-Manipulation
import numpy as np                                                          # Für Arrays
import cv2                                                                  # Import und Export OpenCV
from datetime import datetime                                               # Für das Auslesen des aktuellen Datum und Zeit

def getNumberImages(pImportPath, rows, cols):                               # Funktion: Die Anzahl der Bilder in der Datei bestimmten, Rückgabewert: Anzahl Bilder
    file = open(pImportPath,'rb')                                           # File erneut öffnen, da ansonsten der "Cursor" falsch liegt
    data = np.fromfile(file,dtype=np.uint16)                                # komplettes File einlesen
    numberImages = (np.size(data) - 50) / (rows*cols)                       # Die Anzahl der Bilder bestimmen: die Größe des uint16 Arrays bestimmen, dann minus 50 Elemten (der Header der Datei) und dann geteilt durch die Pixel-Anzahl eines Bildes
    file.close()                                                            # File schließen
    return numberImages                                                     # Die Anzahl der Bilder zurückgeben

def getFarbtiefe(pImportPath):                                              # Funktion: Farbtiefe einer HIS-Datei bestimmten, Rückgabewert: Farbtiefe im Format uint16 oder uint8
    file = open(pImportPath,'rb')                                           # File erneut öffnen, da ansonsten der "Cursor" falsch liegt
    data = np.fromfile(file,dtype=np.uint16)                                # komplettes File einlesen
    cols = int(np.take(data, 8))                                            # Breite auslesen
    rows = int(np.take(data, 9))                                            # Höhe auslesen
    numberImages = int(np.take(data, 10))                                   # Anzahl der Bilder auslesen
    farbtiefe = (np.size(data) - 50) * 16 / (rows * cols * numberImages)    # Die Anzahl der Bilder bestimmen: die Größe des uint16 Arrays bestimmen, dann minus 50 Elemten (der Header der Datei) und dann geteilt durch die Pixel-Anzahl eines Bildes
    file.close()                                                            # File schließen
    if int(farbtiefe) == 16:                                                # 16 Bit pro Pixel?
        tempVar = np.zeros(1,dtype=np.uint16)                               # Für einheitiger Rückgabetyp
        farbtiefe = tempVar.dtype                                           # Für einheitiger Rückgabetyp
    elif int(farbtiefe) == 8:                                               # 8 Bit pro Pixel?
        tempVar = np.zeros(1,dtype=np.uint8)                                # Für einheitiger Rückgabetyp
        farbtiefe = tempVar.dtype                                           # Für einheitiger Rückgabetyp
    else:                                                                   # Keine 8 oder 16 Bit pro Pixel?
        farbtiefe = "FEHLER"                                                # Datei gibt keinen Sinn
        print("Fehler, unbekanntes HIS Format")                             # debug Ausgabe
    return farbtiefe                                                        # Farbtiefe 





def hisImportFunction(pImportPath, pExport = False, pMittelwert = False, pExportPath = "", pZeit=""):                        # Funktion: Bilder im HIS-Format importieren, Übergabewert: Path zum Bild
    aktuelleZeit = pZeit   # aktuelles Datum und Zeit
    pathWithoutExtension = os.path.splitext(pImportPath) [0]                # Pfad ohne Dateiendung erzeugen, .his wird entfernt
    #print("\n\n*************************************************************")
    #print("Funktion zum Einlesen von HIS-Dateien aufgerufen")
    #print("*************************************************************\n")
    fileName, fileExtension = os.path.splitext(os.path.basename(pImportPath))# Dateinamen und Dateiendung extrahieren    
    print("Die Datei", fileName, "wird jetzt eingelesen.")
    fd = open(pImportPath,'rb')                                             # Das Bild öffnen im "rb"-Modus: read binary
    data = np.fromfile(fd,dtype=np.uint16, count=50)                        # Den Header 50 mal mit unsinged int 16 Bit einlesen (erste 100 Bytes)
    rows = int(np.take(data, 8))                                            # Reihen bestimmen, in int konvertieren, ansonsten overflow Error bei der Funktion fromfile()
    cols = int(np.take(data, 9))                                            # Spalten bestimmen
    numberImages = int(getNumberImages(pImportPath, rows, cols))            # Anzahl der Bilder in der Datei bestimmen
    print("Ihre Datei hat", rows, "Reihen und", cols, "Spalten und besteht aus", numberImages, "Bild(ern)")
    bildDaten = np.zeros((numberImages, rows, cols),dtype=np.uint16)                        # leeres 3D-Array der Bilddaten: 1. Nr. des Bildes, 2. Bildhöhe, 3. Bildbreite
    for index in range(numberImages):                                       # Alle Bilder anzeigen und speichern
        f = np.fromfile(fd, dtype=np.uint16, count=rows*cols)               # Pixel lesen und in einem ein dimensionales Array speichern
        im = f.reshape((rows, cols))                                        # Array in zwei dimensionales Array mit rows x cols erstellen
        bildDaten[index] = im                                               # Aktuelles Bild speichern
       #test 
        #backtorgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        #cv2.imshow('image2', backtorgb)
        #cv2.imwrite('asdfasf.png', backtorgb, [cv2.IMWRITE_PNG_COMPRESSION,0])
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #Ende test
        if pExport == True:
            """
            cv2.imshow('image', im)                                             # Array als Bild anzeigen
            cv2.imwrite(pathWithoutExtension+"_"+str(index)+'_beta.png',im, [cv2.IMWRITE_PNG_COMPRESSION,0])     # Array als PNG speichern ohne Kompression
            print("Ihre Datei wurden unter", pathWithoutExtension+"_"+str(index)+".png gespeichert")
            cv2.waitKey()                                                       # Warten bis eine Taste gedrückt wird              
            """
            dirName = os.path.join(pExportPath, "Originalbilder vom " + aktuelleZeit)
            if os.path.exists(dirName):
                pass
            else:
                os.mkdir(dirName)
            fileName = os.path.splitext(os.path.basename(pImportPath))[0] +  "_original_" + str(index) + ".png"
            cv2.imwrite(os.path.join(dirName, fileName), im)    # Array als PNG speichern ohne Kompression
    #if pExport == True:
        #cv2.destroyAllWindows()                                                 # Alle Fenster schließen    
    
    if pMittelwert == True:
        meanImage = np.zeros([rows,cols],dtype=np.uint32)
        for index in range(np.shape(bildDaten)[0]):     # Anzahl der Bilddateien
            meanImage = meanImage + bildDaten[index]
        #meanImage = meanImage / np.shape(bildDaten)[0]
        meanImage = meanImage / numberImages
        meanImage = np.rint(meanImage) # Werte auf Ganzzahlen runden
        ergMeanImage = np.zeros((1, rows, cols), dtype=np.uint16)   # 3D-Array erzeugen, ansonsten nicht kompatibel
        ergMeanImage[0] = np.array(meanImage, dtype=np.uint16)
        #print(ergMeanImage)

        if pExport == True:
            dirName = os.path.join(pExportPath, "Originalbilder vom " + aktuelleZeit)
            if os.path.exists(dirName):
                pass
            else:
                os.mkdir(dirName)
            fileName = os.path.splitext(os.path.basename(pImportPath))[0] +  "_original_mittelwert.png"
            cv2.imwrite(os.path.join(dirName, fileName), ergMeanImage[0])    # Array als PNG speichern ohne Kompression
            """
            cv2.imshow("Mittelwert",ergMeanImage)
            cv2.imwrite(pathWithoutExtension+ "_mittelwert.png",ergMeanImage, [cv2.IMWRITE_PNG_COMPRESSION,0])     # Array als PNG speichern ohne Kompression
            print("Ihre Datei wurden unter", pathWithoutExtension + "_mittelwert.png gespeichert")
            cv2.waitKey()                                                       # Warten bis eine Taste gedrückt wird              
            """
    fd.close()                                                              # File schließen
    if pMittelwert == True:
        return ergMeanImage
    else:
        return bildDaten

def importFunction(pImportPath, pExport = False, pExportPath="", pZeit=""): #vill noch ne fehlermeldung Wenn der Path kein Link enthält!?
    aktuelleZeit = pZeit   # aktuelles Datum und Zeit
    bild = cv2.imread(pImportPath, flags= -1)
    if np.shape(bild) == ():    # wenn das Lesen nicht erfoglreich war
        return ( np.zeros( (1, 512, 512), dtype=np.uint16) ) 
    bildDaten = np.zeros( (1, np.shape(bild)[0], np.shape(bild)[1]), dtype=bild.dtype)
    if np.shape(bildDaten[0]) == np.shape(bild):    # Wenn die Array eine andere Größe besitzen, d.h. wenn es kein Farbbild ist
        bildDaten[0] = bild
    else:
        print("Farbbild")
    #cv2.imshow('image', bildDaten[0])
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    if pExport == True:
        dirName = os.path.join(pExportPath, "Originalbilder vom " + aktuelleZeit)
        if os.path.exists(dirName):
            pass
        else:
            os.mkdir(dirName)
        fileName = os.path.splitext(os.path.basename(pImportPath))[0] +  "_original.png"
        #cv2.imshow('image', bild)                                             # Array als Bild anzeigen
        #cv2.imwrite(os.path.splitext(os.path.basename(pImportPath)) [0] + "importiertesBild.png",bild, [cv2.IMWRITE_PNG_COMPRESSION,0])     # Array als PNG speichern ohne Kompression
        #cv2.imwrite(os.path.join(pExportPath, "Originalbilder", os.path.basename(pImportPath)) + "_Original.png", bild, [cv2.IMWRITE_PNG_COMPRESSION,0])    # Array als PNG speichern ohne Kompression
        cv2.imwrite(os.path.join(dirName, fileName), bild)    # Array als PNG speichern ohne Kompression
        
        #cv2.waitKey()
        #cv2.destroyAllWindows()      
    return bildDaten
   

   
def importUIFunction(pImportPath, pMittelwert = True, pExport = False, pExportPath="", pMittelwertGesamt = False): # Rückgabe Bild-Array und Auflösung Breite und Höhe
    #bildDaten = []
    rows, cols, anzahl, farbtiefe = getAufloesungUndAnzahlUndFarbtiefe(pImportPath[0])
    bildDaten = np.empty((0,rows,cols), dtype=farbtiefe)
    aktuelleZeit = str(datetime.now())[:-7].replace(":","-")   # aktuelles Datum und Zeit
    for aktuellerPfad in pImportPath:
        dateiEndung = (os.path.splitext(os.path.basename(aktuellerPfad)) [1]).lower() # Dateiendung aus dem Pfad seperarieren, kleinschreiben, weil manche OS (Windows) Dateieindungen Groß schreiben
        if dateiEndung == ".his": # Eine his-Datei
            aktuellesArray = hisImportFunction(aktuellerPfad, pExport, pMittelwert, pExportPath, aktuelleZeit)
            bildDaten = np.append(bildDaten, aktuellesArray, axis= 0)
            #bildDaten.append( hisImportFunction(aktuellerPfad, pExport, pMittelwert) )
        elif dateiEndung == ".png" or dateiEndung == ".jpg" or dateiEndung == ".jpeg" or dateiEndung == ".tif" or dateiEndung == ".tiff":
            #bildDaten.append( importFunction(aktuellerPfad, pExport) )
            bildDaten = np.append(bildDaten, importFunction(aktuellerPfad, pExport, pExportPath=pExportPath, pZeit= aktuelleZeit), axis= 0 )
    if pMittelwertGesamt:
        meanImage = np.zeros([rows,cols],dtype=np.uint32)
        for index in range(np.shape(bildDaten)[0]):     # Anzahl der Bilddateien
            meanImage = meanImage + bildDaten[index]
        meanImage = meanImage / np.shape(bildDaten)[0]
        meanImage = np.rint(meanImage) # Werte auf Ganzzahlen runden
        ergMeanImage = np.zeros((rows, cols), dtype=farbtiefe)   # 3D-Array erzeugen, ansonsten nicht kompatibel
        ergMeanImage = np.array(meanImage, dtype=farbtiefe)
        #cv2.imshow("bild.png", ergMeanImage) 
        #cv2.waitKey()  
        return ergMeanImage
    else:
        return bildDaten
   
def getAufloesungUndAnzahlUndFarbtiefe(pImportPath):
    dateiEndung = (os.path.splitext(os.path.basename(pImportPath))[1]).lower() # kleinschreiben, weil manche OS (Windows) Dateieindungen Groß schreiben
    if dateiEndung == ".his":                                                       # Eine his-Datei
        fd = open(pImportPath,'rb')                                             # Das Bild öffnen im "rb"-Modus: read binary
        data = np.fromfile(fd,dtype=np.uint16, count=50)                        # Den Header 50 mal mit unsinged int 16 Bit einlesen (erste 100 Bytes)
        rows = int(np.take(data, 8))                                            # Reihen bestimmen, in int konvertieren, ansonsten overflow Error bei der Funktion fromfile()
        cols = int(np.take(data, 9))                                            # Spalten bestimmen
        anzahl = int(np.take(data, 10))                                         # Anzahl der Bilder aus der HIS Datei auslesen.
        fd.close()                                                              # File schließen
        farbtiefe = getFarbtiefe(pImportPath)
        # anzahl = getNumberImages(pImportPath, rows, cols)                                                     
        # farbtiefe = data.dtype.name
        # farbtiefe = data.dtype
    elif dateiEndung == ".png" or dateiEndung == ".jpg" or dateiEndung == ".jpeg" or dateiEndung == ".tif" or dateiEndung == ".tiff":
        bild = cv2.imread(pImportPath, flags= -1)                               # Bilder mit OpenCV einlesen
        if bild is None:
            print("Fehler beim imread")
            rows = 0
            cols = 0
            anzahl = 0
            tempVar = np.zeros(1,dtype=np.int64)                                # Für einheitiger Rückgabetyp
            farbtiefe = tempVar.dtype                                           # Datei gibt keinen Sinn
        else:
            rows = np.shape(bild)[0]                                                # Reihenanzahl bestimmen 
            cols = np.shape(bild)[1]                                                # Spaltenanzahl bestimmen
            anzahl = 1
            farbtiefe = bild.dtype
    return rows, cols, anzahl, farbtiefe

def checkGreyimage(pImportPath):
    bild = cv2.imread(pImportPath, flags= -1)
    bildDaten = np.zeros( (1, np.shape(bild)[0], np.shape(bild)[1]), dtype=bild.dtype)
    if np.shape(bildDaten) == np.shape(bild):    # Wenn die Array eine andere Größe besitzen, d.h. wenn es kein Farbbild ist
        return True
    else:
        return False   


"""
import os
import platform
import subprocess

def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

open_file("/Users/julian/Documents")
"""


"""
meineListe = 0

meineListe = np.zeros((2,3,3),dtype=np.uint16)
#print("\n-----------------\n", meineListe)
#meineListe[0,0,0]  = 1
#print("\n-----------------\n", meineListe)

meineListe = np.append(meineListe,[[[3, 3, 3 ],[3, 3, 3],[3, 3, 3]]], axis= 0 )


meineListe = np.empty((0,3,3), dtype=np.uint16)
print("\n-----------------\n", meineListe)
meineListe = np.append(meineListe,[[[3, 3, 3 ],[3, 3, 3],[3, 3, 3]]], axis= 0 )
print("\n-----------------\n", meineListe)
meineListe = np.append(meineListe,[[[43,43, 43 ],[43, 43, 43],[43, 43, 43]]], axis= 0 )
print("\n-----------------\n", meineListe)

pass

"""

# deprecated/veraltete Funktionen
"""
def getAufloesungUndAnzahlUndFarbtiefe2(pImportPath):
    dateiEndung = (os.path.splitext(os.path.basename(pImportPath)) [1]).lower() # kleinschreiben, weil manche OS (Windows) Dateieindungen Groß schreiben
    if dateiEndung == ".his": # Eine his-Datei
        fd = open(pImportPath,'rb')                                             # Das Bild öffnen im "rb"-Modus: read binary
        data = np.fromfile(fd,dtype=np.uint16, count=50)                        # Den Header 50 mal mit unsinged int 16 Bit einlesen (erste 100 Bytes)
        rows = int(np.take(data, 8))                                            # Reihen bestimmen, in int konvertieren, ansonsten overflow Error bei der Funktion fromfile()
        cols = int(np.take(data, 9))                                            # Spalten bestimmen
        fd.close()        
        anzahl = getNumberImages(pImportPath, rows, cols)                                                     # File schließen
        #farbtiefe = data.dtype.name
        farbtiefe = data.dtype
    elif dateiEndung == ".png" or dateiEndung == ".jpg" or dateiEndung == ".jpeg" or dateiEndung == ".tif" or dateiEndung == ".tiff":
        bild = cv2.imread(pImportPath, flags= -1)
        rows = np.shape(bild)[0]
        cols = np.shape(bild)[1]
        anzahl = 1
        #farbtiefe = bild.dtype.name
        farbtiefe = bild.dtype
    return rows, cols, anzahl, farbtiefe


def importUIFunctionAlt(pImportPath, pExport = False): # Rückgabe Bild-Array und Auflösung Breite und Höhe
    dateiEndung = (os.path.splitext(os.path.basename(pImportPath)) [1]).lower()
    if dateiEndung == ".his": # Eine his-Datei
        bildDaten = hisImportFunction(pImportPath, pExport)
    elif dateiEndung == ".png" or dateiEndung == ".jpg" or dateiEndung == ".jpeg" or dateiEndung == ".tif" or dateiEndung == ".tiff":
        bildDaten = importFunction(pImportPath, pExport)
    return bildDaten


def hisImportFunction2(pImportPath, pExport = False):                       # Funktion: Bilder im HIS-Format importieren, Übergabewert: Path zum Bild
    pathWithoutExtension = os.path.splitext(pImportPath) [0]                # Pfad ohne Dateiendung erzeugen, .his wird entfernt
    print("\n\n*************************************************************")
    print("Funktion zum Einlesen von HIS-Dateien aufgerufen")
    print("*************************************************************\n")
    fileName, fileExtension = os.path.splitext(os.path.basename(pImportPath))# Dateinamen und Dateiendung extrahieren    
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
            cv2.imwrite(pathWithoutExtension+"_"+str(index)+'_beta.png',im)     # Array als PNG speichern ohne Kompression
            print("Ihre Datei wurden unter", pathWithoutExtension+"_"+str(index)+".png gespeichert")
            cv2.waitKey()                                                       # Warten bis eine Taste gedrückt wird      
    if pExport == True:
        cv2.destroyAllWindows()                                                 # Alle Fenster schließen    
    fd.close()                                                              # File schließen
    return im


"""