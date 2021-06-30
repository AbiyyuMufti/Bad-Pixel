""" 
/*
 * @Author: Andreas Bank
 * @Email: diegruppetg@gmail.com
 * @Date: 15.06.2020
 * @Last Modified by: Andy
 * @Last Modified time: 19.09.20
 * @Description: Die Suchalgorithmen für bad Pixel
 * @Version V2: Digital BPM, Schneller
 */
 """

import config as cfg
import numpy as np
import math

global Lichtschutz
Lichtschutz = 0.02 #10%

# Hot Pixel finder:
def HotPixelFinder(D2_Bild, Schwelle=0.99):
    Zaehler=0
    hohe, breite=np.shape(D2_Bild)
    BPM=np.zeros((hohe,breite))
    for z in range(hohe):
        for s in range(breite):
            if D2_Bild[z,s]>=int(2**  cfg.Farbtiefe)*Schwelle:
                BPM[z,s]=1 #Digital
                Zaehler +=1
    #print("Hot Pixel: " , Zaehler)
    if Zaehler>hohe*breite*Lichtschutz: #Überbelichtungsschutz 
        Zaehler=-1
        #print("Ueberbelichtet")
    return BPM, Zaehler

# Dead Pixel finder:
def DeadPixelFinder(D2_Bild, Schwelle=0.01):
    Zaehler=0
    hohe, breite=np.shape(D2_Bild) 
    BPM=np.zeros((hohe,breite))
    for z in range(hohe):
        for s in range(breite):
            if D2_Bild[z,s]<=int(2**  cfg.Farbtiefe)*Schwelle:
                BPM[z,s]=1 #Digital
                Zaehler +=1
    #print("Tote Pixel: " , Zaehler)
    if Zaehler>hohe*breite*Lichtschutz: 
        Zaehler=-1
        #print("zu viele Dead Pixel")
    return BPM, Zaehler 

def MultiPicturePixelCompare(D3_Bilder,GrenzeHot=0.99,GrenzeDead=0.01):
    Bilderanzahl, hohe, breite=np.shape(D3_Bilder) 
    print(Bilderanzahl," Bilder pruefen...")
    BPM_D=np.zeros((hohe,breite))
    BPM_H=np.zeros((hohe,breite))
    Ungueltig=np.zeros((hohe,breite))
    UberLicht=0
    UnterLicht=0
    for i in range(Bilderanzahl):  
        #print("Bild Nr. ",i)
        cfg.lock.acquire()
        if cfg.killFlagThreads == True: #kill Tread
            cfg.errorCode=-1
            cfg.lock.release() 
            return -6
        cfg.Ladebalken=cfg.Ladebalken+1 
        cfg.lock.release()        
        BPM_Dead, Anz_Dead =DeadPixelFinder(D3_Bilder[i],GrenzeDead) #Check for Black
        if(Anz_Dead<0):
            BPM_Dead=Ungueltig
            UnterLicht=UnterLicht+1
        BPM_D=BPM_D+BPM_Dead
        BPM_Hot, Anz_Hot =HotPixelFinder(D3_Bilder[i],GrenzeHot) #Check HOT
        if(Anz_Hot<0):
            BPM_Hot=Ungueltig
            UberLicht=UberLicht+1
        BPM_H=BPM_H+BPM_Hot
   #Auswertung
    print(UberLicht," Bilder Ueberbelichtet, ",UnterLicht," Bilder zu Dunkel")
    BPM_D=(BPM_D-int(0.3*(Bilderanzahl-UnterLicht)))>0 #Digit + mehr als 30%
    BPM_H=(BPM_H-int(0.3*(Bilderanzahl-UberLicht)))>0
    BPM=np.logical_or(BPM_D,BPM_H)
    Fehler=np.nonzero(BPM)
    Fehler=len(Fehler[0])
    print("Multi Picture findet ",Fehler)
    #Tread Zeugs
    cfg.lock.acquire()
    if UberLicht>Bilderanzahl/3: #Warning
        cfg.errorCode=-3
    cfg.fehlerSammler["MPPC"]=Fehler
    cfg.Global_BPM_Multi =BPM #Tread
    cfg.Ladebalken=cfg.Ladebalken+1 #Final 
    cfg.lock.release()
    return BPM, Fehler 
    
def top(x,Max):
    if x>=Max:
        return Max
    else:
        return x

def bottom(x):
    if x<0:
        return 0
    else:
        return x

def advancedMovingWindow(D3_Bild, Fensterbreite=6, Faktor=3): #Faktor aus Literatur= 3  (BildSerie2 70µA ist Faktor 2,5-3,5)
    Anz, hoehe, breite = np.shape(D3_Bild)
    # hoehe, breite, Anz = np.shape(D3_Bild)
    #print(hoehe,breite)
    BPM=np.zeros((hoehe,breite))
    for i in range(Anz):
        D2_Bild=D3_Bild[i]
        quadrat=int(Fensterbreite/2) #+1
        for y in range(breite):
            if cfg.killFlagThreads == True: #kill Tread / aMW zu langsam für Abbruch nach Bild.
                cfg.errorCode=-1
                return 
            for x in range(hoehe):
                supBPM=D2_Bild[bottom(x-quadrat):top(x+quadrat,breite),bottom(y-quadrat):top(y+quadrat,hoehe)]
                #a= np.shape(supBPM)[0]+1
                #b= np.shape(supBPM)[1]+1
                #Elemente=a+b
                #print("Elemente",Elemente," a=",a," b=",b)
                Std=np.sqrt(np.var(supBPM))
                #debug= abs(np.mean(supBPM)-D2_Bild[x,y])
                if float(Std)*Faktor< abs(np.mean(supBPM)-D2_Bild[x,y]): #TypeError: can't multiply sequence by non-int of type 'numpy.float64'
                    BPM[x,y]=1 #Digital
                    #print("Std: ",Std," Abweichung= ", abs(np.mean(supBPM)-Bilder[Nr,x,y]))
        cfg.lock.acquire()
        cfg.Ladebalken=cfg.Ladebalken+1 
        cfg.lock.release()
    Zaehler=np.count_nonzero(BPM)
    print("advWindow erkennt ",Zaehler," Fehler. Festerbreite= ",Fensterbreite)
    #Tread Zeugs
    cfg.lock.acquire()
    cfg.fehlerSammler["aMW"]=Zaehler
    cfg.Global_BPM_Moving =BPM #Tread
    cfg.Ladebalken=cfg.Ladebalken+1 #Final 
    cfg.lock.release()
    return BPM ,Zaehler 

def dynamicCheck(D3_Bilder, Faktor=1.5): #Bilder müssen verschiene sein (Helle und Dunkle!) , Faktor ist Schwellwert für Erkennung. 1.03-1.2
    Anz, hoehe, breite = np.shape(D3_Bilder)
    if Anz<2:
        print("zu wenig Bilder")
        cfg.lock.acquire()
        cfg.Ladebalken=cfg.Ladebalken+Anz #Damit ist es abgearbeitet.
        cfg.lock.release()
        cfg.errorCode=-2
        return -1 #Hoffeltlich wird das Ausgewertet.
    BPM=np.zeros((hoehe,breite))
    Zaehler=0
    #Hell Dunkel erstellen
    Hellste=np.ones((hoehe,breite))
    Dunkelste=np.ones((hoehe,breite))*2**cfg.Farbtiefe
    for Nr in range(Anz):
        cfg.lock.acquire()
        if cfg.killFlagThreads == True: #kill Tread
            cfg.errorCode=-1
            cfg.lock.release()
            return -6
        cfg.Ladebalken=cfg.Ladebalken+1 
        cfg.lock.release()
        for s in range(hoehe):
            for z in range(breite):
                if Hellste[s,z]<D3_Bilder[Nr,s,z]:
                    Hellste[s,z]=D3_Bilder[Nr,s,z]
                if Dunkelste[s,z]>D3_Bilder[Nr,s,z]:
                    Dunkelste[s,z]=D3_Bilder[Nr,s,z]
    #Mittlere Dynamik
    Dynamik=(Hellste-Dunkelste)/Hellste
    DynamikNorm=np.mean(Dynamik)
    print("Dynamik Norm = ", DynamikNorm)
    for s in range(hoehe):
        for z in range(breite):
            if Dynamik[s,z]<DynamikNorm/Faktor:
                BPM[s,z]=int((DynamikNorm-Dynamik[s,z])/DynamikNorm*100)
                BPM[s,z]=1 #Digital 
                Zaehler+=1
    print(Zaehler," Fehler gefunden (DynamikCheck).")
    #Tread Zeugs
    cfg.lock.acquire()
    cfg.fehlerSammler["dC"]=Zaehler
    cfg.Global_BPM_Dynamik =BPM #Tread
    cfg.Ladebalken=cfg.Ladebalken+1 #Final 
    cfg.lock.release()
    return BPM, Zaehler 

def Mapping(BPM_A,BPM_B,BPM_C=0,BPM_D=0,BPM_E=0):
    BPM=np.logical_or(BPM_A,BPM_B)
    BPM=np.logical_or(BPM,BPM_C)
    BPM=np.logical_or(BPM,BPM_D)
    BPM=np.logical_or(BPM,BPM_E)
    return BPM