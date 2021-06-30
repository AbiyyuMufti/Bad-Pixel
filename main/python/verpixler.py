import random as r
import config as cfg
import numpy as np

def verpixeln (D2_Bild, AnzPixel=100, lineEnable=False,Cluster=False, minAbweichung=20):
    hoehe, breite = np.shape(D2_Bild)   #Bild Dimensionen
    r.seed(D2_Bild[1,1]+AnzPixel)       #Zufallsgenerator 
    BPM=np.zeros((hoehe,breite))        #Leer Bild erstellen
    for i in range(AnzPixel):           
        x=r.randint(0, breite-1)
        y=r.randint(0, hoehe-1)          #Zufall Position
        grey=r.randint(0,2**cfg.Farbtiefe) #Zufall Grauwert
        if(abs(D2_Bild[x,y]-grey)>2**cfg.Farbtiefe*minAbweichung/100):
            D2_Bild[x,y] = grey           #Pixel einfügen
            BPM[x,y] = 100                #Fehler in perfekter BPM vermerken
        else:
            i -=1

    if lineEnable:
        print("Es werden Linienfehler erzeugt")
        for v in range(lineEnable):
            length=r.randint(10,int(breite/3))      #Zufällige Länge der Linie
            dir=r.choice(['X','Y'])                 #Zufällige Richtung
            x=r.randint(0,breite-1)                 #Zufällige Startposition
            y=r.randint(0,hoehe-1)
            if(dir=='X'):
                x=r.randint(0,breite-length)        #Linie vollständig im Bild enthalten
            else:
                y=r.randint(0,breite-length)

            grey=r.randint(0,2**cfg.Farbtiefe)      #Zufall Grauwert
            if(abs(D2_Bild[x,y]-grey)>2**cfg.Farbtiefe*minAbweichung/100):
                if(dir=='X'):
                    D2_Bild[x:x+length,y]=grey
                    BPM[x:x+length,y]=100           #BPM schreiben
                else:
                    D2_Bild[x,y:y+length]=grey
                    BPM[x,y:y+length]=100
            else:
                v -=1
    if Cluster:
        a,b,c=cluster(D2_Bild,Anz=Cluster)
        BPM=BPM+b
        D2_Bild=a
    print("Bad-Pixel wurden erzeugt")
    return D2_Bild, BPM
            

def cluster (D2_Bild, Durchmesser=8, Anz=1, Dichte=1/3):
    print("Cluster wird erzeugt Anzahl ist verfaelscht")
    hoehe, breite = np.shape(D2_Bild)
    r.seed(D2_Bild[1,1]+Anz+Durchmesser)
    BPM=np.zeros((hoehe,breite))
    Zaehler=0
    for i in range(Anz):
        x=r.randint(int(Durchmesser/2),breite-int(Durchmesser/2)-1)
        y=r.randint(int(Durchmesser/2),hoehe-int(Durchmesser/2)-1)
        grey=r.randint(D2_Bild[x,y],2**cfg.Farbtiefe)
        grey=r.randint(grey,2**cfg.Farbtiefe)
        grey=r.randint(grey,2**cfg.Farbtiefe)
        for v in range(int((Durchmesser**2) *Dichte)):
            x2=int(r.gauss(0,Durchmesser/3))
            y2=int(r.gauss(0,Durchmesser/3))
            D2_Bild[x+x2,y+y2]=grey
            BPM[x+x2,y+y2]=100
            Zaehler+=1
        
    return D2_Bild, BPM, Zaehler

def auswertung(BPM_2test, BPM_Original, Namenszusatz='0'):
    if np.shape(BPM_2test) != np.shape(BPM_Original):
        print("Digga Dimensionen!")
        return -1
    hoehe, breite = np.shape(BPM_2test)
    Zaehler=[0,0,0,0,0,0] #True Pos, True Neg, False Pos, False Neg, TPR, FPR
    BadPixelAnz=0
    DetectedAnz=0
    for s in range(hoehe):
        for z in range(breite):
            if(BPM_Original[s,z]>0 and BPM_2test[s,z]>0):
                Zaehler[0]+=1
            elif (BPM_Original[s,z]==0 and BPM_2test[s,z]==0):
                Zaehler[1]+=1
            elif (BPM_Original[s,z]>0 and BPM_2test[s,z]==0):
                Zaehler[2]+=1
            elif (BPM_Original[s,z]==0 and BPM_2test[s,z]>0):
                Zaehler[3]+=1
            if BPM_Original[s,z]:
                BadPixelAnz+=1
            if BPM_2test[s,z]:
                DetectedAnz+=1
    
    NichtErkannt=(1-DetectedAnz/BadPixelAnz)*100 #Prozent
    FalschErkannt=DetectedAnz/(hoehe*breite-BadPixelAnz)*100

    #Plot
    if Zaehler[0]==0:
        Zaehler[4]=1.1
    else:
        Zaehler[4]=Zaehler[0]/(Zaehler[0]+Zaehler[3]) #TPR
    if Zaehler[2]==0:
        Zaehler[5]=0
    else:
        Zaehler[5]=Zaehler[2]/(Zaehler[2]+Zaehler[1]) #FPR

    #np.savetxt("Auswertung_"+Namenszusatz+".csv", Zaehler,fmt="%1.3f")
    print("Auswertung: True Pos, True Neg, False Pos, False Neg")
    print(Zaehler)
    print("Nicht Erkannt = ",NichtErkannt," Falsch Erkannt = ",FalschErkannt )

    return Zaehler, NichtErkannt, FalschErkannt, Zaehler[4:6]

def Julian(Hell, Dunkel, Noch_eins):
    hoehe, breite = np.shape(Hell)
    SchwarzBild=np.zeros((hoehe,breite))
    Zaehler=0
    Fehler,BPM=verpixeln(SchwarzBild,lineEnable=3,Cluster=3, minAbweichung=1)
    for s in range(hoehe):
            for z in range(breite):
                if (BPM[s,z]>0):
                    Hell[s,z]=Fehler[s,z]
                    Dunkel[s,z]=Fehler[s,z]
                    Noch_eins[s,z]=Fehler[s,z]
                    Zaehler+=1
    return Hell, Dunkel, Noch_eins, BPM, Zaehler