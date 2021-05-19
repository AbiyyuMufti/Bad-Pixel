"""
/*
 * @Author: Julian Schweizerhof und Andreas Bank
 * @Email: diegruppetg@gmail.com
 * @Date: 2020-10-16 12:09:24
 * @Last Modified by: JLS666
 * @Last Modified time: 2021-03-05 22:42:34
 * @Description: Main des Projektes, primär ermöglicht diese Datei die GUI
 */
"""

""" Import der Bibliotheken:__________________________________________________________________________________"""
# globale Bibliotheken
from fbs_runtime.application_context.PyQt5 import ApplicationContext    # für das fbs
import sys
import PyQt5.QtCore as core
import PyQt5.QtWidgets as widgets
import PyQt5.QtGui as gui
import PyQt5.uic as uic
import types
import os
import numpy as np
from _thread import start_new_thread, allocate_lock #oder mit therading lib.
import threading
import platform     # für das Öffnen des File Explores
import subprocess   # für das Öffnen des File Explores
from datetime import datetime
import shutil
import copy
import cv2
# lokale Bibliotheken
import importPictures as imP
import exportPictures as exP
import Speichern
import config as cfg
import detection
import correction
import telemetry


""" Beginn der Hauptfunktion:__________________________________________________________________________________"""
if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext    # für das fbs

    """ Globale Variablen:___________________________________________________________________________________ """
    aktuellerTab = 0    # Zustand des Tabs der GUI
    anzahlBilder = 0    # Anzahl der importierten Bilder für die Zeilenanzahl der Tabelle
    anzahlBilderHell = 0    # Anzahl der importierten Bilder Hell für die Zeilenanzahl der Tabelle Flat-Field-Korrektur
    anzahlBilderDunkel = 0    # Anzahl der importierten Bilder Dunkel für die Zeilenanzahl der Tabelle Flat-Field-Korrektur
    sensorList = ["Bitte Ihren Sensor auswählen"]
    bildDaten = 0       # hier werden die importierten Bilder gespeichert, 3D-Array: [anzahlBilder][Zeilen][Spalten]
    bildDatenHell = 0       # hier werden die importierten Bilder gespeichert, 2D-Array: [Zeilen][Spalten]
    bildDatenDunkel = 0       # hier werden die importierten Bilder gespeichert, 2D-Array: [Zeilen][Spalten]
    DATA = 0            # Die Daten für die Speicherung der Config Datei
    mittelwertBilder = 0    # Mittelwert aller importierten Bilder
    flagBPMVorschau = False # Flag für die Anzeige der BPM-Vorschau auf dem ersten Tab
    BAD_Ges = 0 # für das Speichern der BAD Pixel Map

    """ Laden der Gui-UI-Dateien:___________________________________________________________________________________ """
    app = widgets.QApplication(sys.argv)
    dateiMainWindow = appctxt.get_resource("badPixelMainWindow.ui")
    mW = uic.loadUi(dateiMainWindow)        # UI-Fenster MainWindow laden
    dateiNeueBPM = appctxt.get_resource("neueBPM.ui")
    nB = uic.loadUi(dateiNeueBPM)  
    dateiEinstellungSuchen = appctxt.get_resource("einstellungenSuchen.ui")
    eS = uic.loadUi(dateiEinstellungSuchen)  
    dateiEinstellungKorrigieren = appctxt.get_resource("einstellungenKorrigieren.ui")
    eK = uic.loadUi(dateiEinstellungKorrigieren)
    dateiFlatFieldKorrektur = appctxt.get_resource("flatFieldKorrektur.ui")
    fF = uic.loadUi(dateiFlatFieldKorrektur)
    dateiFortschritt = appctxt.get_resource("fortschritt.ui")
    fortschritt = uic.loadUi(dateiFortschritt)
    dateiBildFenster = appctxt.get_resource("bildFenster.ui")
    bildFenster = uic.loadUi(dateiBildFenster)
    msgBox = widgets.QMessageBox()  # Die Message Box
    """ Funktionen für die GUI:___________________________________________________________________________________ """
    ############ Allgemeine Funktionen ########################################################################################
    def startClicked():     # Funktion wenn der Start-Button geklickt wird
        global aktuellerTab
        #Speichern wie bei forward
        Speichern.Write_json(DATA) #Schreiben am Ende   # Julian: eigentlich unnötig, da startClicked einem Forward Klick entspricht
        
        # Check BPM is valid
        if DATA["Sensors"] == [] and mW.checkBoxRohbilderSpeichern.isChecked() == False:    # wenn es keine Sensoren gibt
            openMessageBox(icon=widgets.QMessageBox.Information, text="Sie müssen hierfür zuerst einen Sensor erstellen",informativeText="Für die Suche und der Korrektur muss zuerst ein Sensor erstellt werden.",windowTitle="Sie müssen hierfür zuerst einen Sensor erstellen",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            mW.tabWidget.setCurrentIndex(0)
            return False
        elif DATA["Sensors"] == [] and ( mW.checkBoxAlgorithmusSuchen.isChecked() == True or mW.checkBoxAlgorithmusKorrigieren.isChecked() == True): # wenn etwas außer Rohbilder angehakt wurde und kein Sensor gibt
            openMessageBox(icon=widgets.QMessageBox.Information, text="Sie müssen hierfür zuerst einen Sensor erstellen",informativeText="Für die Suche und der Korrektur muss zuerst ein Sensor erstellt werden.",windowTitle="Sie müssen hierfür zuerst einen Sensor erstellen",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            mW.tabWidget.setCurrentIndex(0)
            return False
        # Check Bilddaten are valid
        if mW.tableWidgetBilddaten.rowCount() == 0:
            openMessageBox(icon=widgets.QMessageBox.Information, text="Keine Bilder importiert",informativeText="Bitte importieren Sie Bilder.",windowTitle="Keine Bilder importiert",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            aktuellerTab = 1
            mW.tabWidget.setCurrentIndex(aktuellerTab)
            return False
        for aktuelleZeile in range(mW.tableWidgetBilddaten.rowCount()):
            # Beim Suchen und Korrigieren wird folgendes überprüft... 
            if mW.checkBoxAlgorithmusSuchen.isChecked() or mW.checkBoxAlgorithmusKorrigieren.isChecked():
                # ...ob die Auflösung der Bilder identisch ist, Rohbilder können auch mit verschieden Auflösungen exportiert werden.
                if mW.tableWidgetBilddaten.item(0, 1).text() != mW.tableWidgetBilddaten.item(aktuelleZeile, 1).text():
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Die Auflösung der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder.",windowTitle="Falsche Auflösung",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    aktuellerTab = 1
                    mW.tabWidget.setCurrentIndex(aktuellerTab)
                    return False
                # ...ob die Farbtiefe der Bilder identisch ist, Rohbilder können auch mit verschieden Farbtiefen exportiert werden.
                if mW.tableWidgetBilddaten.item(0, 3).text() != mW.tableWidgetBilddaten.item(aktuelleZeile, 3).text():
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Die Farbtiefe der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder.",windowTitle="Falsche Farbtiefe",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    aktuellerTab = 1
                    mW.tabWidget.setCurrentIndex(aktuellerTab)
                    return False
                # ...ob die Auflösung der Bilder und der BPM identisch sind, Rohbilder können trotzdem exportiert werden.
                """
                if mW.tableWidgetBilddaten.item(aktuelleZeile, 1).text() != "": # todo: Überprüfen ob die Auflösung der Bilder und der BPM identisch sind, Format: "rows x cols", Beachte: wenn es noch keine gibt, muss es übersprungen werden
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Auslösung der Bad-Pixel-Map und der importierten Bilder sind unterschiedlich",informativeText="Die Auflösung des Bildes aus der " + str(aktuelleZeile) + " Zeile ist nicht mit der Auflösung der Bad-Pixel identisch. Bitte die falschen Bilder löschen oder einen anderen Sensor auswählen.",windowTitle="Falsche Auflösung BPM oder Bilder",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return False  
                """
        # Check Speicherort is valid
        if mW.checkBoxAlgorithmusKorrigieren.isChecked() or mW.checkBoxRohbilderSpeichern.isChecked(): # Speicherort ist beim Suchalgorithmus nicht notwendig. 
            if os.path.isdir(mW.lineEditSpeicherort.text()) == False: #exists(mW.lineEditSpeicherort.text()) == False:
                openMessageBox(icon=widgets.QMessageBox.Information, text="Der eingegebene Pfad für den Speicherort ist nicht gültig",informativeText="Der Pfad: \"" + mW.lineEditSpeicherort.text() + "\" ist kein gültiger Ordnerpfad. Bitte ändern Sie den eingegebenen Pfad.",windowTitle="Kein gültiger Pfad",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                aktuellerTab = 2
                mW.tabWidget.setCurrentIndex(aktuellerTab)
                return False
        # Check Algorithmus is valid
        if mW.checkBoxAlgorithmusSuchen.isChecked() == True:
            if mW.checkBoxAlgorithmusWindow.isChecked() and anzahlBilder>8: #Nur Warnung
                openMessageBox(icon=widgets.QMessageBox.Information, text="Eine große Zahl an Bildern führt zu erhöhten Laufzeiten bei dem Moving-Window-Algorithmus.",informativeText="Wählen Sie andere Algorithmen, oder wenden Sie den Moving-Window-Algorithmus nur auf eine Untermenge der Importe an. Für die Korrektur können anschließend alle Ihrer Importe ohne Suchen verarbeitet werden.",windowTitle="Laufzeitwarnung Moving-Window",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            if mW.checkBoxAlgorithmusDynamic.isChecked() and anzahlBilder<3:
                openMessageBox(icon=widgets.QMessageBox.Information, text="Die Anzahl an Bildern ist zu gering für einen Dynamic Algorithmus",informativeText="Erhöhen Sie die Importe, oder wählen Sie z.B. Moving Window",windowTitle="geringe Anzahl an Bildern Dynamic",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)     
                return False
        # wenn nicht angekreuzt wurde:
        if mW.checkBoxAlgorithmusSuchen.isChecked() == False and mW.checkBoxAlgorithmusKorrigieren.isChecked() == False and mW.checkBoxRohbilderSpeichern.isChecked() == False:
            openMessageBox(icon=widgets.QMessageBox.Information, text="Sie haben nichts ausgewählt",informativeText="Bitte wählen sie mind. eine Checkbox aus.",windowTitle="Nichts ausgewählt.",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            return False
        # wenn nichts innerhalb von Suchen angekreuzt wurde:
        if mW.checkBoxAlgorithmusSuchen.isChecked():
            if mW.checkBoxAlgorithmusSchwellwertfilter.isChecked() == False and mW.checkBoxAlgorithmusDynamic.isChecked() == False and mW.checkBoxAlgorithmusWindow.isChecked() == False:
                openMessageBox(icon=widgets.QMessageBox.Information, text="Sie haben nichts für den Suchalgorithmus ausgewählt",informativeText="Bitte wählen sie mind. eine Checkbox für den Suchalgorithmus aus.",windowTitle="Nichts ausgewählt Suchalgorithmus.",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                return False
        
        # Check Flat-Field-Korrektur ist valid:
        if mW.checkBoxAlgorithmusFFK.isChecked() and mW.checkBoxAlgorithmusKorrigieren.isChecked():
            pass


        # Update UI
        fortschritt.textEdit.clear()    # Info Textfeld löschen
        fortschritt.buttonBox.button(widgets.QDialogButtonBox.Ok).setEnabled(False) # Okay Button disable
        #todo Prüfen ob es bereits alte Biler gibt für die FFK

        # wenn nur Suchen ausgewählt ist, soll nicht der Speicherort anzeigen Button erscheinen
        if mW.checkBoxAlgorithmusSuchen.isChecked() == True and mW.checkBoxAlgorithmusKorrigieren.isChecked() == False and mW.checkBoxRohbilderSpeichern.isChecked() == False:
            fortschritt.pushButtonOeffnen.setVisible(False)
        else:
            fortschritt.pushButtonOeffnen.setVisible(True)
        # Import Pictures
        
        global bildDaten
        global mittelwertBilder
        pathlist = []
        for index in range(anzahlBilder):   # alle Pfade aus der Tabelle in eine Liste schreiben
            pathlist.append(mW.tableWidgetBilddaten.item(index,4).text())
        if mW.checkBoxRohbilderSpeichern.isChecked():
            bildDaten = imP.importUIFunction(pathlist,pMittelwert=True, pExport=True, pExportPath= mW.lineEditSpeicherort.text())
            fortschritt.textEdit.insertPlainText("Rohbilder wurden unter: \"" + mW.lineEditSpeicherort.text() + "\" gespeichert.\n")
        else:
            bildDaten = imP.importUIFunction(pathlist,pMittelwert=True, pExport=False)
        mittelwertBilder = imP.importUIFunction(pImportPath=pathlist,pMittelwertGesamt=True)

        #Import Flatfield Bilder
        if mW.checkBoxAlgorithmusFFK.isChecked() and mW.checkBoxAlgorithmusKorrigieren.isChecked():
            if fF.radioButtonNeueBilder.isChecked():    # neue Bilder werden eingefügt
                global anzahlBilderHell
                global anzahlBilderDunkel
                global bildDatenHell
                global bildDatenDunkel
                pathlistHell = []
                pathlistDunkel = []
                for index in range(anzahlBilderHell):   # alle Pfade aus der Tabelle in eine Liste schreiben
                    pathlistHell.append(fF.tableWidgetHell.item(index,4).text())
                bildDatenHell = imP.importUIFunction(pImportPath=pathlistHell,pMittelwertGesamt=True)
                for index in range(anzahlBilderDunkel):
                    pathlistDunkel.append(fF.tableWidgetDunkel.item(index,4).text())
                bildDatenDunkel = imP.importUIFunction(pImportPath=pathlistDunkel,pMittelwertGesamt=True)
        
                # FFK Bilder speichern
                Speichern.saveFFK(mW.comboBoxBPMSensor.currentText(), bildDatenHell, bildDatenDunkel)
            else:   # Bilder sollen geladen werden
                bildDatenHell, bildDatenDunkel = Speichern.loadFFK(mW.comboBoxBPMSensor.currentText())
                if bildDatenHell == []:
                    print("Fehler")
                if bildDatenDunkel == []:
                    print("Fehler")

            #if( np.shape(imP.importUIFunction(mW.tableWidgetBilddaten.item(index,4).text())) [0] > 1):
            #print(np.shape(imP.importUIFunction(mW.tableWidgetBilddaten.item(index,4).text())) [0])
            #bildDaten.append( imP.importUIFunction(mW.tableWidgetBilddaten.item(index,4).text()) [0] ) #Mehre Bilder gehen nicht...
        #print(np.shape(bildDaten)) 

        #Ladebalken init
        cfg.Ladebalken=0
        if mW.checkBoxAlgorithmusSuchen.isChecked():
            Anz=int(mW.checkBoxAlgorithmusSchwellwertfilter.isChecked())+int(mW.checkBoxAlgorithmusWindow.isChecked())+int(mW.checkBoxAlgorithmusDynamic.isChecked())
        else:
            Anz=0
        cfg.LadebalkenMax=Anz*np.shape(bildDaten)[0]+Anz 
        print("Rechenschritte=",cfg.LadebalkenMax)
        # Suchen Treads
        IDs=[]
        if mW.checkBoxAlgorithmusSuchen.isChecked():
            if(mW.checkBoxAlgorithmusSchwellwertfilter.isChecked()):
                #BPM_Schwellwert=detection.MultiPicturePixelCompare(bildDaten,GrenzeHot=0.995,GrenzeDead=0.1)[0]
                T_ID_MPPC=threading.Thread(name="MPPC",target=detection.MultiPicturePixelCompare,args=(bildDaten,float(eS.labelSchwellwertHot.text()),float(eS.labelSchwellwertDead.text())))
                IDs.append(T_ID_MPPC)
                T_ID_MPPC.start()
            if(mW.checkBoxAlgorithmusDynamic.isChecked()):
                #BPM_Dynamik=detection.dynamicCheck(bildDaten,Faktor=1.03)[0]
                T_ID_dC=threading.Thread(name="dc",target=detection.dynamicCheck,args=(bildDaten,float(eS.labelDynamicSchwellwert.text())))
                IDs.append(T_ID_dC)
                T_ID_dC.start()
            if(mW.checkBoxAlgorithmusWindow.isChecked()):
                #BPM_Window=detection.advancedMovingWindow(bildDaten[0],Faktor=2.0,Fensterbreite=10)[0] 
                T_ID_aMW=threading.Thread(name="aMW",target=detection.advancedMovingWindow,args=(bildDaten,int(eS.labelMovingFensterbreite.text()),float(eS.labelMovingSchwellwert.text())))
                IDs.append(T_ID_aMW)
                T_ID_aMW.start()
        #====Jetzt wird gesucht!====#
        timer.start(cfg.recallTime) # ms heruntersetzen für Performance
        """
        pixmap = gui.QPixmap(" ")
        
        fortschritt.label.setPixmap(pixmap)
        fortschritt.label.setScaledContents(True)
        fortschritt.label.resize(pixmap.width(), pixmap.height())
        """

        fortschritt.progressBar.setValue(0)
        if fortschritt.exec() == widgets.QDialog.Rejected: #Abbrechen
            print("Gecancelt gedrueckt") # hier muss dann der Prozess gestoppt werden. 
            cfg.killFlagThreads=True #alle Treads killen
            cfg.Ladebalken=0
            timer.stop() #Prozess ist damit abgeschalten.
            print("Try to join")
            for ID in IDs:
                if ID.is_alive():
                    ID.join()
                    print(ID,"der leuft ja noch!")
            print("Treads sind alle tot")
            cfg.killFlagThreads=False
            cv2.destroyAllWindows()
        else:
            if mW.checkBoxAlgorithmusSuchen.isChecked():    # nur eine neue Bad Pixelmap speichern, wenn auch Fehler gesucht wurden
                Speichern.BPM_Save(BAD_Ges*150,mW.comboBoxBPMSensor.currentText()) #BPM Speichern
            cv2.destroyAllWindows()
            updateBPM() 
            updateTextBPM() # Text auf dem erstem Tab aktualisieren
            showBPM()
        print("startClicked")   # Debug
    def msgButtonClick():
        print("message")
        pass
    def openMessageBox(icon, text, informativeText, windowTitle, standardButtons, pFunction):
        msgBox.setIcon(icon) # msgBox.setIcon(widgets.QMessageBox.Information)
        msgBox.setText(text) # msgBox.setText("Der eingegebene Pfad für den Speicherort ist nicht gültig")
        msgBox.setInformativeText(informativeText) # msgBox.setInformativeText("Der Pfad: \"" + mW.lineEditSpeicherort.text() + "\" ist kein gültiger Pfad. Bitte ändern Sie den eingegebenen Pfad.")
        msgBox.setWindowTitle(windowTitle) #msgBox.setWindowTitle("Kein gültiger Pfad")
        msgBox.setStandardButtons(standardButtons) #msgBox.setStandardButtons(widgets.QMessageBox.Ok) # | widgets.QMessageBox.Cancel)
        msgBox.buttonClicked.connect(pFunction) # msgBox.buttonClicked.connect(msgButtonClick)
        returnValue = msgBox.exec()
        if returnValue == widgets.QMessageBox.Ok:
            print("OK clicked") # Debug
        return returnValue        
    def mW_pushButtonMainBack():
        global aktuellerTab     # ohne diese Zeile kommt darunter eine Fehlermeldung
        if aktuellerTab > 0:
            aktuellerTab = aktuellerTab - 1
        mW.tabWidget.setCurrentIndex(aktuellerTab)
        if aktuellerTab < 3:
            mW.pushButtonMainForward.setText("Weiter")
        if aktuellerTab <= 0:
            mW.pushButtonMainBack.setVisible(False)
        # print(aktuellerTab)   # Debug      
    def mW_pushButtonMainForward():
        global aktuellerTab     # ohne diese Zeile kommt darunter eine Fehlermeldung
        if aktuellerTab == 0:
            global flagBPMVorschau
            cv2.destroyAllWindows()
            flagBPMVorschau = False
            mW.pushButtonBPMVorschau.setText("BPM-Vorschau an")
        if aktuellerTab >= 3:
            startClicked()
            return
        if aktuellerTab < 3:
            aktuellerTab = aktuellerTab + 1 
        mW.tabWidget.setCurrentIndex(aktuellerTab)
        if aktuellerTab >= 3:
            mW.pushButtonMainForward.setText("Start")
        if aktuellerTab > 0:
            mW.pushButtonMainBack.setVisible(True)
        # print(aktuellerTab)   # debug
        Speichern.Write_json(DATA) #Schreiben am Ende
    def mW_tabWidget():
        global aktuellerTab
        aktuellerTab = mW.tabWidget.currentIndex()
        if aktuellerTab != 0:   # BPM Vorschau deaktivieren
            global flagBPMVorschau
            cv2.destroyAllWindows()
            flagBPMVorschau = False
            mW.pushButtonBPMVorschau.setText("BPM-Vorschau an")
        # print(aktuellerTab)   # debug
        if aktuellerTab >= 3:
            mW.pushButtonMainForward.setText("Start")
        if aktuellerTab > 0:
            mW.pushButtonMainBack.setVisible(True)
        if aktuellerTab < 3:
            mW.pushButtonMainForward.setText("Weiter")
        if aktuellerTab <= 0:
            mW.pushButtonMainBack.setVisible(False)
    def uiSetup():  # alles was beim Laden passieren soll
        # Aktuelle Tab speichern
        global aktuellerTab
        global DATA
        global sensorList
        aktuellerTab = mW.tabWidget.currentIndex()
        DATA=Speichern.Read_json() #Lesen zu Beginn #-1 Abfangen?!
        # Tab-Widget
        if aktuellerTab >= 3:
            mW.pushButtonMainForward.setText("Start")
        if aktuellerTab > 0:
            mW.pushButtonMainBack.setVisible(True)
        if aktuellerTab < 3:
            mW.pushButtonMainForward.setText("Weiter")
        if aktuellerTab <= 0:
            mW.pushButtonMainBack.setVisible(False)
        # Tab: Sensor/BPM
        sensorList=Speichern.WelcheSensorenGibtEs(DATA)[1]
        mW.comboBoxBPMSensor.clear()
        mW.comboBoxBPMSensor.addItems(sensorList)
        mW.comboBoxBPMSensor.setCurrentText(DATA["last_GenutzterSensor"])
        updateBPM()     # Comboboxes aktualisieren
        updateTextBPM() # Text auf dem erstem Tab aktualisieren
        showBPM()
        # Tab: Algorithmus - GroupBox Pixelfehler finden enablen
        if mW.checkBoxAlgorithmusSuchen.isChecked():
            mW.groupBoxSuchen.setEnabled(True)
            # print("Checked")  # debug
        else:
            mW.groupBoxSuchen.setEnabled(False)
            # print("not Checked")  # debug
        # Tab: Algorithmus - GroupBox Pixelfehler korrigieren enablen
        if mW.checkBoxAlgorithmusKorrigieren.isChecked():
            mW.groupBoxKorrigieren.setEnabled(True)
            # print("Checked")  # debug
        else:
            mW.groupBoxKorrigieren.setEnabled(False)
            # print("not Checked") # debug
        mW.checkBoxBilddaten.setVisible(False) # noch nicht implementiert
        # Werte Suchen Laden
        if DATA["Sensors"] == []:
            print("Keine Sensoren") # debug
        else:
            eS.horizontalSliderSchwellwertHot.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_oben"])
            eS.horizontalSliderSchwellwertDead.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_unten"])
            eS.horizontalSliderMovingFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fensterbreite_advWindow"])
            eS.horizontalSliderMovingSchwellwert.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_advWindow"])
            eS.horizontalSliderDynamicSchwellwert.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_Dynamik"])
        # Einstellungen Suchen
        eS_horizontalSliderSchwellwertHot()
        eS_horizontalSliderSchwellwertDead()
        eS_horizontalSliderMovingFensterbreite()
        eS_horizontalSliderMovingSchwellwert()
        eS_horizontalSliderDynamicSchwellwert()
        eS.line_3.setVisible(False)             # nicht implementier
        eS.pushButtonVorschau.setVisible(False) # nicht implementier
        # Einstellungen Korrigieren
        eK_horizontalSliderNachbarFensterbreite()
        eK_horizontalSliderGradientFensterbreite()
        eK.line_3.setVisible(False)             # nicht implementier
        eK.pushButtonVorschau.setVisible(False) # nicht implementier
        # Werte Korrekur Laden
        if DATA["Sensors"] == []:
            print("Keine Sensoren") # debug
        else:
            eK.horizontalSliderNachbarFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Nachbar"])
            eK.horizontalSliderGradientFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Gradient"])
        # Fortschritt Fenster
        fortschritt.buttonBox.button(widgets.QDialogButtonBox.Ok).setEnabled(False) # Okay Button disable

    def updateBPM():
        mW.comboBoxBPMBPM.clear()
        bpmList=Speichern.WelcheBPMGibtEs(mW.comboBoxBPMSensor.currentText())
        mW.comboBoxBPMBPM.addItems(bpmList)
        if len(bpmList) <= 0:
            setEnabledBPM(False)
        else:
            setEnabledBPM(True)
        

    def updateTextBPM():
        mW.textEditBPM.clear()
        if platform.system() == "Windows":
            mW.textEditBPM.setFontPointSize(11)
        else:
            mW.textEditBPM.setFontPointSize(15)
        if mW.comboBoxBPMSensor.currentText() == "":    # wenn es keinen Sensor gibt
            print("Es gibt kein Sensor!")
            setEnabledBPM(False)
            mW.textEditBPM.insertPlainText("Herzlich willkomen zum Bad-Pixel-Programm \n\n")
            mW.textEditBPM.insertPlainText("Sie haben noch keinen Sensor erstellt.\n")
            mW.textEditBPM.insertPlainText("Drücken Sie hierfür bitte den Knopf \"Neuer Sensor erstellen ...\"\n")
            mW.textEditBPM.insertPlainText("Außerdem können Sie üben den Knopf \"Sensoren laden ...\" bereits erstellte Sensoren importieren.\n\n")
            mW.textEditBPM.insertPlainText("Viel Spaß mit dem Programm wünschen Andreas B. und Julian S.")
            setEnabledSensor(False)
            return
        setEnabledSensor(True)
        lokalBPM=Speichern.BPM_Read(mW.comboBoxBPMSensor.currentText())
        aufloesung = np.shape(lokalBPM)
        #print("Rueckgabe aufloesung: ", aufloesung) # Diese Zeile macht ein bug unter mac OS, Programm öffnet und schlißet sich sofort wieder. Lag an dem ü
        if aufloesung == ():  # noch keine BBM vorhanden
            if platform.system() == "Windows":
                mW.textEditBPM.insertPlainText("Name des Sensors:\t\t\t" + DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Sensor_Name"] + "\n")
                mW.textEditBPM.insertPlainText("\nEs wurde noch keine Pixelfehler-Liste (Bad-Pixel-Map) angelegt.")
            else:
                mW.textEditBPM.insertPlainText("Name des Sensors:\t\t" + DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Sensor_Name"] + "\n")
                mW.textEditBPM.insertPlainText("\nEs wurde noch keine Pixelfehler-Liste (Bad-Pixel-Map) angelegt.")
        else:
            zeilen, spalten = aufloesung
            
            geleseneBilder = DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Anz_Bilder"]
            #anzahlPixelfehler = DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Anz_PixelFehler"]
            anzahlPixelfehler = Speichern.getFehleranzahlBPM(mW.comboBoxBPMBPM.currentText())
            mW.textEditBPM.insertPlainText("Name des Sensors:\t\t" + DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Sensor_Name"] + "\n")
            mW.textEditBPM.insertPlainText("Sensor Auflösung:\t\t" + str(zeilen) + " x " + str(spalten) + "\n")
            mW.textEditBPM.insertPlainText("Sensor Erstelldatum:\t\t" +  str(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Erstell_Datum"]) + "\n")
            mW.textEditBPM.insertPlainText("Gelesene Bilder des Sensors:\t" + str(geleseneBilder) + "\n")
            mW.textEditBPM.insertPlainText("Anzahl Pixelfehler:\t\t" + str(anzahlPixelfehler) + "\n")
            mW.textEditBPM.insertPlainText("Anteil Pixelfehler:\t\t" + str( round(anzahlPixelfehler/(spalten * zeilen)*100, 2)) + " % \n")
            mW.textEditBPM.insertPlainText("Letzte Änderung:\t\t" +  str( Speichern.getModTimeBPM(mW.comboBoxBPMBPM.currentText()) ) + "\n")
    def showBPM():
        global flagBPMVorschau
        if flagBPMVorschau == True:
            cv2.destroyAllWindows()
            if mW.comboBoxBPMBPM.count() != 0:     # wenn's eine BPM gibt diese auch anzeigen
                akutelleBPM = Speichern.BPM_Read_Selected(mW.comboBoxBPMBPM.currentText())
                akutelleBPM = akutelleBPM.astype(np.uint8) # weil es sonst ein 16 Bit Bild ist
                textWindow =  "Bad-Pixel-Map von " + mW.comboBoxBPMBPM.currentText()
                cv2.imshow(textWindow, akutelleBPM)
            else:   # Wenns noch keine gibt.
                print("keine BPM")  
                openMessageBox(icon=widgets.QMessageBox.Information, text="Es gibt noch keine Bad-Pixel-Map (BPM) für diesen Sensor",informativeText="Sie müssen erst Pixelfehler suchen, damit eine BPM erstellt wird.",windowTitle="Keine BPM vorhanden!",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)

    def setEnabledBPM(flag):
        mW.labelBPMchoose.setEnabled(flag)
        #mW.labelBPMChoose.setText(core.QStandardPaths.writableLocation(core.QStandardPaths.AppDataLocation))
        mW.comboBoxBPMBPM.setEnabled(flag)
    
    def setEnabledSensor(flag):
        mW.labelSensorChoose.setEnabled(flag)
        #mW.labelBPMChoose.setText(core.QStandardPaths.writableLocation(core.QStandardPaths.AppDataLocation))
        mW.comboBoxBPMSensor.setEnabled(flag)
  

    def openFFKWindow():
        global anzahlBilderHell
        global anzahlBilderDunkel
        if fF.exec() == widgets.QDialog.Accepted:
            cv2.destroyAllWindows()
            if fF.radioButtonNeueBilder.isChecked():    # wenn neue Bilder gesucht werden
                if anzahlBilderHell == 0:
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Sie haben keine Hellbilder importiert ",informativeText="Bitte importieren Sie Hellbilder",windowTitle="Nichts importiert",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    openFFKWindow()
                    return False
                if anzahlBilderDunkel == 0:
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Sie haben keine Dunkelbilder importiert ",informativeText="Bitte importieren Sie Dunkelbilder",windowTitle="Nichts importiert",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    openFFKWindow()
                    return False  
                for aktuelleZeile in range(fF.tableWidgetHell.rowCount()):
                    # ...ob die Auflösung der Bilder identisch ist, Rohbilder können auch mit verschieden Auflösungen exportiert werden.
                    if fF.tableWidgetHell.item(0, 1).text() != fF.tableWidgetHell.item(aktuelleZeile, 1).text():
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Die Auflösung der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder bei den Hellbildern von der Flatfield-Korrektur.",windowTitle="Falsche Auflösung",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False
                    # ...ob die Farbtiefe der Bilder identisch ist, Rohbilder können auch mit verschieden Farbtiefen exportiert werden.
                    if fF.tableWidgetHell.item(0, 3).text() != fF.tableWidgetHell.item(aktuelleZeile, 3).text():
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Die Farbtiefe der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder bei den Hellbildern von der Flatfield-Korrektur.",windowTitle="Falsche Farbtiefe",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False
                    # ...ob die Auflösung der Bilder und der BPM identisch sind, Rohbilder können trotzdem exportiert werden.
                    """
                    if fF.tableWidgetHell.item(aktuelleZeile, 1).text() != "": # todo: Überprüfen ob die Auflösung der Bilder und der BPM identisch sind, Format: "rows x cols", Beachte: wenn es noch keine gibt, muss es übersprungen werden
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Auslösung der Bad-Pixel-Map und der importierten Bilder sind unterschiedlich",informativeText="Die Auflösung des Bildes aus der " + str(aktuelleZeile) + " Zeile ist nicht mit der Auflösung der Bad-Pixel identisch. Bitte die falschen Bilder bei den Hellbildern von der Flatfield-Korrektur löschen oder einen anderen Sensor auswählen.",windowTitle="Falsche Auflösung BPM oder Bilder",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False  
                    """
                    if fF.tableWidgetHell.item(aktuelleZeile, 1).text() != fF.tableWidgetDunkel.item(0, 1).text():
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Die Auflöung der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder aus der Flatfield-Korrektur.",windowTitle="Falsche Auslösung",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False
                    if fF.tableWidgetHell.item(aktuelleZeile, 3).text() != fF.tableWidgetDunkel.item(0, 3).text():
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Die Farbtiefe der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder aus der Flatfield-Korrektur.",windowTitle="Falsche Farbtiefe",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False  
                
                for aktuelleZeile in range(fF.tableWidgetDunkel.rowCount()):
                    # ...ob die Auflösung der Bilder identisch ist, Rohbilder können auch mit verschieden Auflösungen exportiert werden.
                    if fF.tableWidgetDunkel.item(0, 1).text() != fF.tableWidgetDunkel.item(aktuelleZeile, 1).text():
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Die Auflösung der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder bei den Dunkelbildern von der Flatfield-Korrektur.",windowTitle="Falsche Auflösung",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False
                    # ...ob die Farbtiefe der Bilder identisch ist, Rohbilder können auch mit verschieden Farbtiefen exportiert werden.
                    if fF.tableWidgetDunkel.item(0, 3).text() != fF.tableWidgetDunkel.item(aktuelleZeile, 3).text():
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Die Farbtiefe der importierten Bilder ist unterschiedlich",informativeText="Bitte entfernen Sie die falschen Bilder bei den Dunkelbildern von der Flatfield-Korrektur.",windowTitle="Falsche Farbtiefe",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False
                    # ...ob die Auflösung der Bilder und der BPM identisch sind, Rohbilder können trotzdem exportiert werden.
                    """
                    if fF.tableWidgetDunkel.item(aktuelleZeile, 1).text() != "": # todo: Überprüfen ob die Auflösung der Bilder und der BPM identisch sind, Format: "rows x cols", Beachte: wenn es noch keine gibt, muss es übersprungen werden
                        openMessageBox(icon=widgets.QMessageBox.Information, text="Auslösung der Bad-Pixel-Map und der importierten Bilder sind unterschiedlich",informativeText="Die Auflösung des Bildes aus der " + str(aktuelleZeile) + " Zeile ist nicht mit der Auflösung der Bad-Pixel identisch. Bitte die falschen Bilder bei den Dunkelbildern von der Flatfield-Korrektur löschen oder einen anderen Sensor auswählen.",windowTitle="Falsche Auflösung BPM oder Bilder",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                        openFFKWindow()
                        return False  
                    """
            #print("Laeuft")
        else:   # wenn Abbrechen geklickt wird
            cv2.destroyAllWindows()
            mW.checkBoxAlgorithmusFFK.setChecked(False)
            # alle Tabellen sollen gelöscht werden
            fF.tableWidgetHell.setRowCount(0)
            anzahlBilderHell = 0
            fF.tableWidgetDunkel.setRowCount(0)
            anzahlBilderDunkel = 0
    def updateFFK():
        if fF.radioButtonNeueBilder.isChecked():    # wenn neue Bilder gewählt ist
            fF.groupBox.setEnabled(True)
            fF.pushButtonGespeicherteBilder.setEnabled(False)
        else:   # wenn alte Bilder geladen werden
            fF.groupBox.setEnabled(False)
            fF.pushButtonGespeicherteBilder.setEnabled(True)

    ############ Ende Allgemeine Funktionen ########################################################################################
    #### ######## Funktionen von dem ab Sensor / BPM ########################################################################################
    
    
    def mW_comboBoxBPMSensor():     # wird aufgerufen, wenn ein neues Element ausgewählt wird
        #print("mW_comboBoxBPMSensor") # debug
        DATA["last_GenutzterSensor"]=mW.comboBoxBPMSensor.currentText()
        updateBPM()
        updateTextBPM()
        showBPM()
    def mW_comboBoxBPMBPM():
        updateTextBPM()
        showBPM()
    def mW_pushButtonBPMNeuerSensor():
        # Ordner auswählen: getExistingDirectory(), Datei auswählen: getOpenFileName(), Dateien auswählen: filename = widgets.QFileDialog.getOpenFileNames() [0]      
        # filename = widgets.QFileDialog.getOpenFileNames() [0]      
        # filename = widgets.QFileDialog.getOpenFileNames(directory = core.QStandardPaths.writableLocation(core.QStandardPaths.DocumentsLocation), filter = "UI-Dateien (*.ui)")
        # mW.lineEditBPM.setText(filename)
        #print("Ordnerdialog geöffnet", filename)
        if nB.exec() == widgets.QDialog.Accepted:
            global sensorList
            tempSensor = [tempSensor.lower() for tempSensor in sensorList] 
            if nB.lineEditNeueBPM.text().lower() in tempSensor:
                print("So einen Sensor gibt es schon")   # debug
                openMessageBox(icon=widgets.QMessageBox.Information, text="Achtung diesen Sensor gibt es schon!", informativeText="Es gibt bereits einen Sensor mit dem selben Namen. Bitte wählen Sie einen anderen Namen für den neuen Sensor.", windowTitle="Achtung diesen Sensor gibt es schon!",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                return
            #sensorList.append(nB.lineEditNeueBPM.text())
            Speichern.SensorAnlegen(nB.lineEditNeueBPM.text(), DATA)
            sensorList=Speichern.WelcheSensorenGibtEs(DATA)[1]
            #mW.comboBoxBPMSensor.clear()
            #mW.comboBoxBPMSensor.addItems(sensorList)
            mW.comboBoxBPMSensor.addItem(sensorList[-1])    # -1 letzes Elemt 
            mW.comboBoxBPMSensor.setCurrentIndex( len(sensorList) - 1) # -1 da Informatiker ab 0 zählen
            DATA["last_GenutzterSensor"]=mW.comboBoxBPMSensor.currentText()
            updateBPM()
            updateTextBPM()
            showBPM()
            Speichern.Write_json(DATA) #Schreiben am Ende
        print("NeueBPM geöffnet")   # debug
    def mW_pushButtonBPMSensorLaden():
        dirname = widgets.QFileDialog.getExistingDirectory()
        if dirname == "":  # wenn  auf abbrechen gedrückt wird
            return
        backValue = widgets.QMessageBox.Ok  # für das untere If
        if mW.comboBoxBPMSensor.currentText() != "":   # wenn es keinen Sensor gibt, soll die Message box nicht angzeigt werden
            backValue = openMessageBox(icon=widgets.QMessageBox.Information, text="Achtung Ihre alten Sensoren werden überschrieben.", informativeText="Wenn sie Ihre alten Sensoren weiterhin behalten wollen, müssen Sie diese erst exportieren. Ist dies der Fall, hier auf Abbrechen oder Cancel klicken. ", windowTitle="Achtung Ihre alten Sensoren werden überschrieben.",standardButtons=widgets.QMessageBox.Ok | widgets.QMessageBox.Cancel,pFunction=msgButtonClick)
        if backValue == widgets.QMessageBox.Ok:
            # Hier Andy Sachen laden
            #Kopieren(dirname,Speichern.dir_path) #ist ein Einzeiler
            files = os.listdir(Speichern.dir_path)
            for aktuellesFile in files:
                os.remove(os.path.join(Speichern.dir_path, aktuellesFile))
            files = os.listdir(dirname)
            for aktuellesFile in files:
                path = os.path.join(dirname, aktuellesFile)
                shutil.copy(path,Speichern.dir_path)
            uiSetup()
            Speichern.Write_json(DATA) #Schreiben am Ende
        print(dirname)
    def mW_pushButtonBPMSensorSpeichern():
        dirname = widgets.QFileDialog.getExistingDirectory()
        if dirname == "":  # wenn  auf abbrechen gedrückt wird
            return  # Funktion verlassen
        # hier wird der Ordner  kopiert
        aktuelleZeit = str(datetime.now())[:-7].replace(":","-")    # aktuelle Zeit speichern
        destination = shutil.copytree(Speichern.dir_path, os.path.join(dirname, "Bad-Pixel-Map " + aktuelleZeit))  # Weil der Zielordner nicht existieren darf
        openMessageBox(icon=widgets.QMessageBox.Information, text="Ihre Daten wurden erfolgreich gespeichert.", informativeText="Die Daten wurden unter " + destination + " gespeichert.", windowTitle="Speichern erfolgreich",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
        # print(dirname)    # debug
    def mW_pushButtonBPMVorschau():
        global flagBPMVorschau
        if flagBPMVorschau == False:
            mW.pushButtonBPMVorschau.setText("BPM-Vorschau aus")
            flagBPMVorschau = True
            showBPM()
        else:
            mW.pushButtonBPMVorschau.setText("BPM-Vorschau an")
            cv2.destroyAllWindows()
            flagBPMVorschau = False

    def mW_pushButtonBPMSensorLoeschen():
        aktuellerIndex = mW.comboBoxBPMSensor.currentIndex()
        currentText = mW.comboBoxBPMSensor.currentText()
        # print(aktuellerIndex) # debug
        if currentText == "": # es gibt kein Sensor
            openMessageBox(icon=widgets.QMessageBox.Information, text="Es gibt keine Sensoren",informativeText="Es gibt keinen Sensor der gelöscht werden kann.",windowTitle="Es gibt keine Sensoren",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
        else:
            backValue = openMessageBox(icon=widgets.QMessageBox.Information, text="Achtung der Sensor und alle dessen BPM werden gelöscht", informativeText="Möchten Sie den Sensor und alle dessen BPM löschen, dann drücken Sie bitte \"OK\" ", windowTitle="Achtung der Sensor und alle dessen BPM werden gelöscht",standardButtons=widgets.QMessageBox.Ok | widgets.QMessageBox.Cancel,pFunction=msgButtonClick)
            if backValue == widgets.QMessageBox.Ok:
                del sensorList[aktuellerIndex]
                mW.comboBoxBPMSensor.removeItem(aktuellerIndex)
                Speichern.SensorLoschen(currentText,DATA)
                Speichern.deleteAllBPM(currentText)
                updateBPM()
                updateTextBPM()
                showBPM()
                Speichern.Write_json(DATA) #Schreiben am Ende
    def mW_pushButtonBPMBPMLoeschen():
        if mW.comboBoxBPMBPM.currentText() == "":    # wenn es keine BPM gibt
            openMessageBox(icon=widgets.QMessageBox.Information, text="Es gibt keine BPM zum Löschen", informativeText="Sie müssen hierfür erst mit einem Sensor eine BPM erzeugen.", windowTitle="Es gibt keine BPM zum Löschen",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            return
        Speichern.deleteBPM(mW.comboBoxBPMBPM.currentText())
        updateBPM()
        updateTextBPM()
        showBPM()
        Speichern.Write_json(DATA) #Schreiben am Ende
    ### Tab Bilddaten
    def mW_pushButtonBilddatenOrdnerDurchsuchen():   # Ordner importieren
        if DATA["Import_Pfad"]==" ":
            dirname = widgets.QFileDialog.getExistingDirectory(directory = core.QStandardPaths.writableLocation(core.QStandardPaths.DocumentsLocation))
        else:
            dirname = widgets.QFileDialog.getExistingDirectory(directory = DATA["Import_Pfad"])
        if dirname != "":  # wenn nicht auf abbrechen gedrückt wird
            mW.lineEditBilddatenDurchsuchen.setText(dirname)
            print(os.listdir(dirname))
        else:
            print("Abgebbrochen")
        # Sind die Daten valide? Umlaute usw. nur für Windows erforderlich
        if platform.system() == "Windows":     
            tempPath = dirname.lower()
            if tempPath.find("ä") != -1:
                #print("Es befindet sich ein ä im Datei- oder Pfadname") # debug
                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ä im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + dirname, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                return
            if tempPath.find("ö") != -1:
                #print("Es befindet sich ein ö im Datei- oder Pfadname") # debug
                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ö im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + dirname,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                return
            if tempPath.find("ü") != -1:
                #print("Es befindet sich ein ü im Datei- oder Pfadname") # debug
                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ü im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + dirname,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                return
            if tempPath.find("ß") != -1:
                #print("Es befindet sich ein ß im Datei- oder Pfadname") # debug
                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ß im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad: ist " + dirname,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                return
        # Valide Daten check zu Ende
        print("Ordnerdialog Bilddaten geöffnet", dirname)
    def mW_pushButtonBilddatenImportieren():
        global anzahlBilder
        dirname = str(mW.lineEditBilddatenDurchsuchen.text())
        if os.path.isdir(dirname):   #os.path.exists(dirname): # wenn der Pfad überhaupt existiert
            if mW.checkBoxBilddaten.isChecked():   # Unterordner auch importieren noch nicht implementiert  
                print("Unterordner werden auch importiert")
            else:   # keine Unterordner importieren
                #if dirname != "": 
                files = os.listdir(dirname) # bug, wenn dirname kein bekannter Pfad ist --> behoben
                print(files,type(files))
                #anzahlBilderLocal = 0
                imageFiles = []
                for aktuellesFile in files:
                    dateiEndung = (os.path.splitext(aktuellesFile) [1]).lower() # lower für Windos
                    if dateiEndung == ".png" or dateiEndung == ".jpg" or dateiEndung == ".jpeg" or dateiEndung == ".tif" or dateiEndung == ".tiff" or dateiEndung == ".his":
                        #anzahlBilderLocal = anzahlBilderLocal + 1
                        imageFiles.append(aktuellesFile)

                        # Sind die Daten valide? Umlaute usw. nur für Windows erforderlich
                        if platform.system() == "Windows": 
                            tempPath = aktuellesFile.lower()
                            if tempPath.find("ä") != -1:
                                #print("Es befindet sich ein ä im Datei- oder Pfadname") # debug
                                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ä im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + dirname + "/" + aktuellesFile, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                                return
                            if tempPath.find("ö") != -1:
                                #print("Es befindet sich ein ö im Datei- oder Pfadname") # debug
                                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ö im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + dirname + "/" + aktuellesFile, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                                return
                            if tempPath.find("ü") != -1:
                                #print("Es befindet sich ein ü im Datei- oder Pfadname") # debug
                                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ü im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + dirname + "/" + aktuellesFile, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                                return
                            if tempPath.find("ß") != -1:
                                #print("Es befindet sich ein ß im Datei- oder Pfadname") # debug
                                openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ß im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad: ist " + dirname + "/" + aktuellesFile, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                                return
                        # Valide Daten check zu Ende
                        
                if len(imageFiles) <= 0:
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Keine Bilder im aktuellen Verzeichnis",informativeText="Das Verzeichnis: \"" + dirname + "\" enthält keine gültigen Bilddateien. Es sind nur Bilder im PNG, JPG, TIF und HIS Format kompatibel. Bitte ändern Sie den eingegebenen Pfad.",windowTitle="Keine Bilder im Verzeichnis",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                anzahlBilder = anzahlBilder + len(imageFiles)
                mW.tableWidgetBilddaten.setRowCount(anzahlBilder) # Soviele Zeilen in der Tabelle aktivieren, wie es Bilder gibt.
                for index in range(len(imageFiles)):  # Alle importieren Bilder durchgehen
                    mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(imageFiles))) ,0, widgets.QTableWidgetItem(imageFiles[index]) ) # Den Dateinamen aller markierten Bilder in die erste Spalte schreiben
                    print(dirname + "/" + imageFiles[index])
                    rows, cols, anzahlHisBilder, farbtiefe = imP.getAufloesungUndAnzahlUndFarbtiefe(os.path.join(dirname,imageFiles[index]))
                    mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(imageFiles))) ,1, widgets.QTableWidgetItem( str(rows) + " x " + str(cols) ) )# Die Auflösung aller markierten Bilder in die erste Spalte schreiben
                    mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(imageFiles))) ,2, widgets.QTableWidgetItem( str( int(anzahlHisBilder)) ) )
                    if type(farbtiefe) == str:
                        mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(imageFiles))) ,3, widgets.QTableWidgetItem(  farbtiefe  ) )
                    else:
                        mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(imageFiles))) ,3, widgets.QTableWidgetItem(  farbtiefe.name ) )
                    mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(imageFiles))) ,4, widgets.QTableWidgetItem( str(os.path.join(dirname,imageFiles[index])) ) )# Die Pfade aller Bilder in die dritten Spalte schreiben
                    # zentrieren
                    mW.tableWidgetBilddaten.item((index + (anzahlBilder - len(imageFiles))), 1).setTextAlignment(core.Qt.AlignCenter)
                    mW.tableWidgetBilddaten.item((index + (anzahlBilder - len(imageFiles))), 2).setTextAlignment(core.Qt.AlignCenter)
                    mW.tableWidgetBilddaten.item((index + (anzahlBilder - len(imageFiles))), 3).setTextAlignment(core.Qt.AlignCenter)
                #else:
                #print("Abgebbrochen")
                print("Keine Unterordner importieren")
                #imP.importUIFunction(mW.tableWidgetBilddaten.item(0,4).text(),True)
                #exP.exportPictures(mW.lineEditSpeicherort.text(), mW.tableWidgetBilddaten.item(0,0).text(),GOOD)
            #Pfad Speichern
            DATA["Import_Pfad"]=dirname
        else:
            openMessageBox(icon=widgets.QMessageBox.Information, text="Der eingegebene Pfad ist nicht gültig",informativeText="Der Pfad: \"" + dirname + "\" ist kein gültiger Ordnerpfad. Bitte ändern Sie den eingegebenen Pfad.",windowTitle="Kein gültiger Pfad",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
        
            
    def mW_pushButtonBilddatenAdd():    # Bilddateien importieren
        global anzahlBilder # globale Variable Anzahl der Bilder bekannt machen
        # file Dialog, kompatible Dateien: *.his *.png *.jpg *.jpeg *.tif *.tiff,
        # Alle Pfäde der Dateien werden in filename gespeichert
        #filename = widgets.QFileDialog.getOpenFileNames(directory = core.QStandardPaths.writableLocation(core.QStandardPaths.DocumentsLocation), filter = "Bild-Dateien (*.his *.png *.jpg *.jpeg *.tif *.tiff)") [0]
        filename = widgets.QFileDialog.getOpenFileNames(filter = "Bild-Dateien (*.his *.png *.jpg *.jpeg *.tif *.tiff)") [0] # directory wird vom OS gewählt
        # print(filename) # debug

        # Sind die Daten valide? Umlaute usw. nur für Windows erforderlich
        if platform.system() == "Windows":
            for file in filename:      
                tempPath = file.lower()
                if tempPath.find("ä") != -1:
                    #print("Es befindet sich ein ä im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ä im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ö") != -1:
                    #print("Es befindet sich ein ö im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ö im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ü") != -1:
                    #print("Es befindet sich ein ü im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ü im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ß") != -1:
                    #print("Es befindet sich ein ß im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ß im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad: ist " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
        # Valide Daten check zu Ende

        anzahlBilder = anzahlBilder + len(filename) # Anzahl der Bilder aktualisieren
        mW.tableWidgetBilddaten.setRowCount(anzahlBilder) # Soviele Zeilen in der Tabelle aktivieren, wie es Bilder gibt.
        # print(anzahlBilder) # debug
        for index in range(len(filename)):  # Alle importieren Bilder durchgehen
            mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(filename))) ,0, widgets.QTableWidgetItem( os.path.basename(filename[index]))) # Den Dateinamen aller markierten Bilder in die erste Spalte schreiben
            rows, cols, anzahlHisBilder, farbtiefe = imP.getAufloesungUndAnzahlUndFarbtiefe(filename[index])
            # data = imP.importUIFunction(filename[index])
            # print(data, imP.np.shape(data))
            mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(filename))) ,1, widgets.QTableWidgetItem( str(rows) + " x " + str(cols) ) )# Die Auflösung aller markierten Bilder in die erste Spalte schreiben
            mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(filename))) ,2, widgets.QTableWidgetItem( str( int(anzahlHisBilder)) ) )
            if type(farbtiefe) == str:
                mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(filename))) ,3, widgets.QTableWidgetItem(  farbtiefe  ) )
            else:
                mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(filename))) ,3, widgets.QTableWidgetItem(  farbtiefe.name ) )
            mW.tableWidgetBilddaten.setItem( (index + (anzahlBilder - len(filename))) ,4, widgets.QTableWidgetItem( str(filename[index]) ) )# Die Pfade aller Bilder in die dritten Spalte schreiben
            mW.tableWidgetBilddaten.item((index + (anzahlBilder - len(filename))), 1).setTextAlignment(core.Qt.AlignCenter)
            mW.tableWidgetBilddaten.item((index + (anzahlBilder - len(filename))), 2).setTextAlignment(core.Qt.AlignCenter)
            mW.tableWidgetBilddaten.item((index + (anzahlBilder - len(filename))), 3).setTextAlignment(core.Qt.AlignCenter)
        #print(os.path.basename(filename[0]))
        #imP.importUIFunction(filename[0])
        #if imP.checkGreyimage(filename[0]):        ##### bug geht nicht bei HIS-Dateien
        #    print("Graubild")
        #else:
        #    print("Farbbild")
        #imP.importUIFunction(mW.tableWidgetBilddaten.item(0,4).text(),True)
        # print("mW_pushButtonBilddatenAdd")    # debug
    
    def mW_pushButtonBilddatenDelete():
        global anzahlBilder
        #print(mW.tableWidgetBilddaten.item(0,0).text())
        zeilen =  mW.tableWidgetBilddaten.selectedItems()
        print(zeilen)
        zeilenLoeschen = []
        alterWert = -1
        for index in zeilen:
            if alterWert != index.row():
                zeilenLoeschen.append( index.row() )
            alterWert = index.row()
            
        print(zeilenLoeschen)
        """
        for index in range(len(zeilen)):
            zeilenLoeschen.append( zeilen[index].row() )
        """
        zeilenLoeschen.sort()
        print(zeilenLoeschen)
        zeilenLoeschen.reverse()
        print(zeilenLoeschen)
        for index in range(len(zeilenLoeschen)):
            mW.tableWidgetBilddaten.removeRow(zeilenLoeschen[index]) 
        anzahlBilder = anzahlBilder - len(zeilenLoeschen)
        print(anzahlBilder)
        
        """
        print(mW.tableWidgetBilddaten.selectedIndexes())
        print(mW.tableWidgetBilddaten.selectedIndexes())
        print(mW.tableWidgetBilddaten.selectedRanges())
        #mW.tableWidgetBilddaten
        #core.QItemSelectionModel.reset()
        """
        
        """
        print(mW.tableWidgetBilddaten())

        widgets.QTableWidget

        mW.tableWidgetBilddaten.setRowCount(2)
        mW.tableWidgetBilddaten.setItem(0,0, widgets.QTableWidgetItem("Yeay"))
        mW.tableWidgetBilddaten.setItem(0,1, widgets.QTableWidgetItem("Yeay2"))
        mW.tableWidgetBilddaten.setItem(1,0, widgets.QTableWidgetItem("Yeay3"))
        mW.tableWidgetBilddaten.setItem(1,1, widgets.QTableWidgetItem("Yeay4"))
        
        mW.tableWidgetBilddaten.setItem(1,0, abc[0])
        mW.tableWidgetBilddaten.setItem(1,1, abc[1])
        """
        #mW.tableWidgetBilddaten.setItem(1,1, widgets.QTableWidgetItem("Yeay4"))
        #print("mW_pushButtonBilddatenDelete")
        
    def mW_pushButtonBilddatenDeleteAll():
        #mW.tableWidgetBilddaten.clearContents()
        #mW.tableWidgetBilddaten.removeRow()
        mW.tableWidgetBilddaten.setRowCount(0)
        #mW.tableWidgetBilddaten.setRowCount(1)
        global anzahlBilder
        anzahlBilder = 0
        #print("mW_pushButtonBilddatenDeleteAll")
    ### Tab Speicherort
    def mW_pushButtonSpeicherortDurchsuchen():
        if DATA["Export_Pfad"]==" ":
            filename = widgets.QFileDialog.getExistingDirectory(directory = core.QStandardPaths.writableLocation(core.QStandardPaths.DocumentsLocation))    
        else:
            filename = widgets.QFileDialog.getExistingDirectory(directory = DATA["Export_Pfad"])    
        mW.lineEditSpeicherort.setText(filename)
        DATA["Export_Pfad"]=filename
        #print("mW_pushButtonSpeicherortDurchsuchen")
    ### Tab Algorithmus
    def mW_checkBoxAlgorithmusSuchen():
        if mW.checkBoxAlgorithmusSuchen.isChecked():
            mW.groupBoxSuchen.setEnabled(True)
            #print("Checked")
        else:
            mW.groupBoxSuchen.setEnabled(False)
            #print("not Checked")
        #print("Suchen")
    def mW_checkBoxAlgorithmusKorrigieren():
        if mW.checkBoxAlgorithmusKorrigieren.isChecked():
            mW.groupBoxKorrigieren.setEnabled(True)
            #print("Checked")
        else:
            mW.groupBoxKorrigieren.setEnabled(False)
            #print("not Checked")
    def mW_pushButtonAlgorithmusSuchenEinstellungen():
        if DATA["Sensors"] == []:    # wenn es keine Sensoren gibt
            openMessageBox(icon=widgets.QMessageBox.Information, text="Sie müssen hierfür zuerst einen Sensor erstellen",informativeText="Für die Einstellungen muss zuerst ein Sensor erstellt werden.",windowTitle="Sie müssen hierfür zuerst einen Sensor erstellen",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            mW.tabWidget.setCurrentIndex(0)
            return
        #Laden
        eS.horizontalSliderSchwellwertHot.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_oben"])
        eS.horizontalSliderSchwellwertDead.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_unten"])
        eS.horizontalSliderMovingFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fensterbreite_advWindow"])
        eS.horizontalSliderMovingSchwellwert.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_advWindow"])
        eS.horizontalSliderDynamicSchwellwert.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_Dynamik"])
        if eS.exec() == widgets.QDialog.Accepted:
            #Speichern
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_oben"]=eS.horizontalSliderSchwellwertHot.value()
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_unten"]=eS.horizontalSliderSchwellwertDead.value()            
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fensterbreite_advWindow"]=eS.horizontalSliderMovingFensterbreite.value()
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_advWindow"]=eS.horizontalSliderMovingSchwellwert.value()
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_Dynamik"]= eS.horizontalSliderDynamicSchwellwert.value()
        else:   # Die alten Werte wiederherstellen, da auf Abbrechnen geklickt wurde
            eS.horizontalSliderSchwellwertHot.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_oben"])
            eS.horizontalSliderSchwellwertDead.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Schwellwert_unten"])
            eS.horizontalSliderMovingFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fensterbreite_advWindow"])
            eS.horizontalSliderMovingSchwellwert.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_advWindow"])
            eS.horizontalSliderDynamicSchwellwert.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Faktor_Dynamik"])
    def mW_pushButtonAlgorithmusKorrigierenEinstellungen():
        if DATA["Sensors"] == []:    # wenn es keine Sensoren gibt
            openMessageBox(icon=widgets.QMessageBox.Information, text="Sie müssen hierfür zuerst einen Sensor erstellen",informativeText="Für die Einstellungen muss zuerst ein Sensor erstellt werden.",windowTitle="Sie müssen hierfür zuerst einen Sensor erstellen",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
            mW.tabWidget.setCurrentIndex(0)
            return
        eK.horizontalSliderNachbarFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Nachbar"])
        eK.horizontalSliderGradientFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Gradient"])
        Methode=DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_korrekturmethode"]
        eK.radioButtonMedian.setChecked(Methode==cfg.Methoden.NMFC.value)
        eK.radioButtonMittelwert.setChecked(Methode==cfg.Methoden.NARC.value)
        eK.radioButtonReplacement.setChecked(Methode==cfg.Methoden.NSRC.value)
        if eK.exec() == widgets.QDialog.Accepted:
            if eK.radioButtonMedian.isChecked():
                Methode=cfg.Methoden.NMFC
            elif eK.radioButtonMittelwert.isChecked():
                Methode=cfg.Methoden.NARC
            else:
                Methode=cfg.Methoden.NSRC
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Nachbar"]=eK.horizontalSliderNachbarFensterbreite.value()
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Gradient"]=eK.horizontalSliderGradientFensterbreite.value()
            DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_korrekturmethode"]=int(Methode.value)
        else:   # Die alten Werte wiederherstellen, da auf Abbrechnen geklickt wurde
            eK.horizontalSliderNachbarFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Nachbar"])
            eK.horizontalSliderGradientFensterbreite.setValue(DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_Fenster_Gradient"])
            eK.radioButtonMedian.setChecked(Methode==cfg.Methoden.NMFC)
            eK.radioButtonMittelwert.setChecked(Methode==cfg.Methoden.NARC)
            eK.radioButtonReplacement.setChecked(Methode==cfg.Methoden.NSRC)
    

    def mW_checkBoxAlgorithmusFFK():
        if mW.checkBoxAlgorithmusFFK.isChecked():
            hellbild, dunkelbild =  Speichern.loadFFK(mW.comboBoxBPMSensor.currentText())
            if (hellbild == []) or (dunkelbild == []):  # wenn es einer von beiden Bildern nicht gibt
                alteFFKBilder = False
            else:   # wenn es alte Bilder von dem Sensor gibt
                alteFFKBilder = True 
            if alteFFKBilder:
                fF.radioButtonGespeicherteBilder.setEnabled(True)
                fF.radioButtonGespeicherteBilder.setChecked(True)
                updateFFK()
            else:
                fF.radioButtonGespeicherteBilder.setEnabled(False)
                fF.radioButtonNeueBilder.setChecked(True)
                updateFFK()
            openFFKWindow()
                
    ### Flat-Field-Korrektur

    def fF_radioButtonGespeicherteBilder():
        updateFFK()
        #print("Radio Button FFK Dunkel")
    def fF_pushButtonHellAdd():
        #print("fF_pushButtonHellAdd")
        global anzahlBilderHell # globale Variable Anzahl der Bilder bekannt machen
        # file Dialog, kompatible Dateien: *.his *.png *.jpg *.jpeg *.tif *.tiff,
        # Alle Pfäde der Dateien werden in filename gespeichert
        filename = widgets.QFileDialog.getOpenFileNames(directory = DATA["Import_Pfad"], filter = "Bild-Dateien (*.his *.png *.jpg *.jpeg *.tif *.tiff)") [0]
        # print(filename) # debug

        # Sind die Daten valide? Umlaute usw. nur für Windows erforderlich
        if platform.system() == "Windows":
            for file in filename:      
                tempPath = file.lower()
                if tempPath.find("ä") != -1:
                    #print("Es befindet sich ein ä im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ä im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ö") != -1:
                    #print("Es befindet sich ein ö im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ö im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ü") != -1:
                    #print("Es befindet sich ein ü im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ü im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ß") != -1:
                    #print("Es befindet sich ein ß im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ß im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad: ist " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
        # Valide Daten check zu Ende

        anzahlBilderHell = anzahlBilderHell + len(filename) # Anzahl der Bilder aktualisieren
        fF.tableWidgetHell.setRowCount(anzahlBilderHell) # Soviele Zeilen in der Tabelle aktivieren, wie es Bilder gibt.
        # print(anzahlBilderHell) # debug
        for index in range(len(filename)):  # Alle importieren Bilder durchgehen
            fF.tableWidgetHell.setItem( (index + (anzahlBilderHell - len(filename))) ,0, widgets.QTableWidgetItem( os.path.basename(filename[index]))) # Den Dateinamen aller markierten Bilder in die erste Spalte schreiben
            rows, cols, anzahlHisBilder, farbtiefe = imP.getAufloesungUndAnzahlUndFarbtiefe(filename[index])
            # data = imP.importUIFunction(filename[index])

            # print(data, imP.np.shape(data))
            fF.tableWidgetHell.setItem( (index + (anzahlBilderHell - len(filename))) ,1, widgets.QTableWidgetItem( str(rows) + " x " + str(cols) ) )# Die Auflösung aller markierten Bilder in die erste Spalte schreiben
            fF.tableWidgetHell.setItem( (index + (anzahlBilderHell - len(filename))) ,2, widgets.QTableWidgetItem( str( int(anzahlHisBilder)) ) )
            if type(farbtiefe) == str:
                fF.tableWidgetHell.setItem( (index + (anzahlBilderHell - len(filename))) ,3, widgets.QTableWidgetItem(  farbtiefe  ) )
            else:
                fF.tableWidgetHell.setItem( (index + (anzahlBilderHell - len(filename))) ,3, widgets.QTableWidgetItem(  farbtiefe.name ) )
            fF.tableWidgetHell.setItem( (index + (anzahlBilderHell - len(filename))) ,4, widgets.QTableWidgetItem( str(filename[index]) ) )# Die Pfade aller Bilder in die dritten Spalte schreiben
            
            fF.tableWidgetHell.item((index + (anzahlBilderHell - len(filename))), 1).setTextAlignment(core.Qt.AlignCenter)
            fF.tableWidgetHell.item((index + (anzahlBilderHell - len(filename))), 2).setTextAlignment(core.Qt.AlignCenter)
            #print(fF.tableWidgetHell.item((index + (anzahlBilderHell - len(filename))), 3))
            fF.tableWidgetHell.item((index + (anzahlBilderHell - len(filename))), 3).setTextAlignment(core.Qt.AlignCenter)
            
    def fF_pushButtonHellDelete():
        global anzahlBilderHell
        zeilen =  fF.tableWidgetHell.selectedItems()
        print(zeilen)
        zeilenLoeschen = []
        alterWert = -1
        for index in zeilen:    # weiß nicht was das macht
            if alterWert != index.row():
                zeilenLoeschen.append( index.row() )
            alterWert = index.row()
        #print(zeilenLoeschen)
        zeilenLoeschen.sort()
        #print(zeilenLoeschen)
        zeilenLoeschen.reverse()
        #print(zeilenLoeschen)
        for index in range(len(zeilenLoeschen)):
            fF.tableWidgetHell.removeRow(zeilenLoeschen[index]) 
        anzahlBilderHell = anzahlBilderHell - len(zeilenLoeschen)
        #print(anzahlBilder)
    def fF_pushButtonHellDeleteAll():
        fF.tableWidgetHell.setRowCount(0)
        global anzahlBilderHell
        anzahlBilderHell = 0
        
    def fF_pushButtonDunkelAdd():
        #print("fF_pushButtonDunkelAdd")
        global anzahlBilderDunkel # globale Variable Anzahl der Bilder bekannt machen
        # file Dialog, kompatible Dateien: *.his *.png *.jpg *.jpeg *.tif *.tiff,
        # Alle Pfäde der Dateien werden in filename gespeichert
        filename = widgets.QFileDialog.getOpenFileNames(directory = DATA["Import_Pfad"], filter = "Bild-Dateien (*.his *.png *.jpg *.jpeg *.tif *.tiff)") [0]
        
        # Sind die Daten valide? Umlaute usw. nur für Windows erforderlich
        if platform.system() == "Windows":
            for file in filename:      
                tempPath = file.lower()
                if tempPath.find("ä") != -1:
                    #print("Es befindet sich ein ä im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ä im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file, windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ö") != -1:
                    #print("Es befindet sich ein ö im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ö im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ü") != -1:
                    #print("Es befindet sich ein ü im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ü im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad ist: " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
                if tempPath.find("ß") != -1:
                    #print("Es befindet sich ein ß im Datei- oder Pfadname") # debug
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Es befindet sich ein ß im Datei- oder Pfadname",informativeText="Dieses Programm kann leider nicht mit Datei- oder Pfadnamen mit Umlauten arbeiten.  Bitte bennen Sie denjenigen Ordner oder Datei um.  Ihr importierter Pfad: ist " + file,windowTitle="Umlaut im Datei- oder Pfadname",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    return
        # Valide Daten check zu Ende
        
        # print(filename) # debug
        anzahlBilderDunkel = anzahlBilderDunkel + len(filename) # Anzahl der Bilder aktualisieren
        fF.tableWidgetDunkel.setRowCount(anzahlBilderDunkel) # Soviele Zeilen in der Tabelle aktivieren, wie es Bilder gibt.
        # print(anzahlBilderDunkel) # debug
        for index in range(len(filename)):  # Alle importieren Bilder durchgehen
            fF.tableWidgetDunkel.setItem( (index + (anzahlBilderDunkel - len(filename))) ,0, widgets.QTableWidgetItem( os.path.basename(filename[index]))) # Den Dateinamen aller markierten Bilder in die erste Spalte schreiben
            rows, cols, anzahlHisBilder, farbtiefe = imP.getAufloesungUndAnzahlUndFarbtiefe(filename[index])
            # data = imP.importUIFunction(filename[index])
            # print(data, imP.np.shape(data))
            fF.tableWidgetDunkel.setItem( (index + (anzahlBilderDunkel - len(filename))) ,1, widgets.QTableWidgetItem( str(rows) + " x " + str(cols) ) )# Die Auflösung aller markierten Bilder in die erste Spalte schreiben
            fF.tableWidgetDunkel.setItem( (index + (anzahlBilderDunkel - len(filename))) ,2, widgets.QTableWidgetItem( str( int(anzahlHisBilder)) ) )
            if type(farbtiefe) == str:
                fF.tableWidgetDunkel.setItem( (index + (anzahlBilderDunkel - len(filename))) ,3, widgets.QTableWidgetItem(  farbtiefe  ) )
            else:
                fF.tableWidgetDunkel.setItem( (index + (anzahlBilderDunkel - len(filename))) ,3, widgets.QTableWidgetItem(  farbtiefe.name ) )
            fF.tableWidgetDunkel.setItem( (index + (anzahlBilderDunkel - len(filename))) ,4, widgets.QTableWidgetItem( str(filename[index]) ) )# Die Pfade aller Bilder in die dritten Spalte schreiben
            
            fF.tableWidgetDunkel.item((index + (anzahlBilderDunkel - len(filename))), 1).setTextAlignment(core.Qt.AlignCenter)
            fF.tableWidgetDunkel.item((index + (anzahlBilderDunkel - len(filename))), 2).setTextAlignment(core.Qt.AlignCenter)
            fF.tableWidgetDunkel.item((index + (anzahlBilderDunkel - len(filename))), 3).setTextAlignment(core.Qt.AlignCenter)
    def fF_pushButtonDunkelDelete():
        global anzahlBilderDunkel
        zeilen =  fF.tableWidgetDunkel.selectedItems()
        print(zeilen)
        zeilenLoeschen = []
        alterWert = -1
        for index in zeilen:    # weiß nicht was das macht
            if alterWert != index.row():
                zeilenLoeschen.append( index.row() )
            alterWert = index.row()
        #print(zeilenLoeschen)
        zeilenLoeschen.sort()
        #print(zeilenLoeschen)
        zeilenLoeschen.reverse()
        #print(zeilenLoeschen)
        for index in range(len(zeilenLoeschen)):
            fF.tableWidgetDunkel.removeRow(zeilenLoeschen[index]) 
        anzahlBilderDunkel = anzahlBilderDunkel - len(zeilenLoeschen)
        #print(anzahlBilder)
    def fF_pushButtonDunkelDeleteAll():
        fF.tableWidgetDunkel.setRowCount(0)
        global anzahlBilderDunkel
        anzahlBilderDunkel = 0
    def fF_radioButtonNeueBilder():
        updateFFK()
        #print("Radio Button FFK Hell")
    def fF_pushButtonGespeicherteBilder():
        Speichern.loadFFK(mW.comboBoxBPMSensor.currentText(),flagShow= True)
    ### Einstellungen Suchen
                                                        #Andy Vorgabe Multi: Hell: min 1 max 0,95 ival 0,002 Dunkel: min 0 max 0,05 ival 0,005 # früher: ival 0,01
                                                        #Andy Vorgabe Moving Fenster: min 5 max 17 ival 2  Faktor: min 2 max 3,5 ival 0,1
                                                        #Andy Vorgabe Dynamic Empfindlichkeit: min 1.03 max 2 ival 0.01
    eS.horizontalSliderSchwellwertHot.setMinimum(0) 
    eS.horizontalSliderSchwellwertHot.setMaximum(50) #Vill etwas mehr.
    eS.horizontalSliderSchwellwertHot.setTickInterval(1)
    def eS_horizontalSliderSchwellwertHot():
        value = eS.horizontalSliderSchwellwertHot.value()
        eS.labelSchwellwertHot.setText( str(round( value*(-0.002)+1,3) ) )

    eS.horizontalSliderSchwellwertDead.setMinimum(0) 
    eS.horizontalSliderSchwellwertDead.setMaximum(50) #Vill etwas mehr.
    eS.horizontalSliderSchwellwertDead.setTickInterval(1)
    def eS_horizontalSliderSchwellwertDead():
        value = eS.horizontalSliderSchwellwertDead.value()
        eS.labelSchwellwertDead.setText( str(round(value*0.002,3) ) ) 

    eS.horizontalSliderMovingFensterbreite.setMinimum(0) 
    eS.horizontalSliderMovingFensterbreite.setMaximum(6)
    eS.horizontalSliderMovingFensterbreite.setTickInterval(1)
    def eS_horizontalSliderMovingFensterbreite():
        value = eS.horizontalSliderMovingFensterbreite.value()
        eS.labelMovingFensterbreite.setText( str(value*2 + 5) )

    eS.horizontalSliderMovingSchwellwert.setMinimum(0) 
    eS.horizontalSliderMovingSchwellwert.setMaximum(15)
    eS.horizontalSliderMovingSchwellwert.setTickInterval(1)
    def eS_horizontalSliderMovingSchwellwert():
        value = eS.horizontalSliderMovingSchwellwert.value()
        eS.labelMovingSchwellwert.setText( str( round(value*(-0.1) + 3.5, 2 ) ) )

    eS.horizontalSliderDynamicSchwellwert.setMinimum(0) 
    eS.horizontalSliderDynamicSchwellwert.setMaximum(97)
    eS.horizontalSliderDynamicSchwellwert.setTickInterval(1)
    def eS_horizontalSliderDynamicSchwellwert():
        value = eS.horizontalSliderDynamicSchwellwert.value()
        eS.labelDynamicSchwellwert.setText( str( round(value*(-0.01) + 2,2)  ) )     

    def eS_pushButtonVorschau():#Detection #Beim Drücken soll eine Vorschau von Bild Nr 1 mit Aktuellen Einstellungen entstehen.
        """
        pixmap = gui.QPixmap("Bild.png")
        #pixmap.scaled()
        bildFenster.label.setPixmap(pixmap)
        bildFenster.label.setScaledContents(True)
        bildFenster.label.resize(pixmap.width(), pixmap.height())
        """
        bildFenster.exec()
        #Aufrugf der Vorschaufunktion/Prozess. /Eigentlich wie Startbutton blos ohne Speichern.
    ### Einstellungen Korrektur 
    eK.horizontalSliderNachbarFensterbreite.setMinimum(0) #Andy Vorgabe: min 3 max 21 ival 2 
    eK.horizontalSliderNachbarFensterbreite.setMaximum(9)
    eK.horizontalSliderNachbarFensterbreite.setTickInterval(2)
    def eK_horizontalSliderNachbarFensterbreite():
        value = eK.horizontalSliderNachbarFensterbreite.value()
        eK.labelNachbarFensterbreite.setText(str((value*2)+3))  # 3, 5, 7, 9 ... 21
        #if value > 9:
        #    eK.labelNachbarFensterbreite.setStyleSheet('color: red')
        #print("eK_horizontalSliderNachbarFensterbreite", value)   # debug
    eK.horizontalSliderGradientFensterbreite.setMinimum(0) #Andy Vorgabe: min 4 max 24 ival 2 
    eK.horizontalSliderGradientFensterbreite.setMaximum(10) #länge des Gradienten
    eK.horizontalSliderGradientFensterbreite.setTickInterval(2)
    def eK_horizontalSliderGradientFensterbreite():
        value = eK.horizontalSliderGradientFensterbreite.value()
        eK.labelGradientFensterbreite.setText(str((value*2)+4))  # 4, 6, 8 ... 22
        #if value > 10:
        #    eK.labelGradientFensterbreite.setStyleSheet('color: red')
        #else:
        #    eK.labelGradientFensterbreite.setStyleSheet('color: black')
        #print("eK_horizontalSliderGradientFensterbreite", value)   # debug

    def eK_pushButtonVorschau():
        """
        pixmap = gui.QPixmap("Bild.png")
        bildFenster.label.setPixmap(pixmap)
        bildFenster.label.setScaledContents(True)
        bildFenster.label.resize(pixmap.width(), pixmap.height())
        """
        bildFenster.exec()
    ### Fortschritt Fenster
    def fortschritt_pushButtonOeffnen():
        path = mW.lineEditSpeicherort.text()
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        #print("Speicherort im File Explorer oeffnen")   # debug

    """ GUI-Elemente mit Funktionen verbinden:___________________________________________________________________________________ """   
    #### UI Aktionen #### 
    ### Allgemein
    uiSetup()
    mW.pushButtonMainBack.clicked.connect(mW_pushButtonMainBack)
    mW.pushButtonMainForward.clicked.connect(mW_pushButtonMainForward)
    mW.tabWidget.currentChanged.connect(mW_tabWidget)

    ### Tab Sensor / BPM
    mW.comboBoxBPMSensor.activated.connect(mW_comboBoxBPMSensor)
    mW.comboBoxBPMBPM.activated.connect(mW_comboBoxBPMBPM)
    mW.pushButtonBPMNeuerSensor.clicked.connect(mW_pushButtonBPMNeuerSensor)
    mW.pushButtonBPMSensorLaden.clicked.connect(mW_pushButtonBPMSensorLaden)
    mW.pushButtonBPMSensorSpeichern.clicked.connect(mW_pushButtonBPMSensorSpeichern)
    mW.pushButtonBPMVorschau.clicked.connect(mW_pushButtonBPMVorschau)
    #mW.pushButtonBPMVorschau.released.connect(mW_pushButtonBPMVorschau_released)
    mW.pushButtonBPMSensorLoeschen.clicked.connect(mW_pushButtonBPMSensorLoeschen)
    mW.pushButtonBPMBPMLoeschen.clicked.connect(mW_pushButtonBPMBPMLoeschen)

    ### Tab Bilddaten
    mW.pushButtonBilddatenOrdnerDurchsuchen.clicked.connect(mW_pushButtonBilddatenOrdnerDurchsuchen)
    mW.pushButtonBilddatenImportieren.clicked.connect(mW_pushButtonBilddatenImportieren)
    mW.pushButtonBilddatenAdd.clicked.connect(mW_pushButtonBilddatenAdd)
    mW.pushButtonBilddatenDelete.clicked.connect(mW_pushButtonBilddatenDelete)
    mW.pushButtonBilddatenDeleteAll.clicked.connect(mW_pushButtonBilddatenDeleteAll)

    ### Tab Speicherort
    mW.pushButtonSpeicherortDurchsuchen.clicked.connect(mW_pushButtonSpeicherortDurchsuchen)

    ### Tab Algorithmus
    mW.checkBoxAlgorithmusSuchen.stateChanged.connect(mW_checkBoxAlgorithmusSuchen)
    mW.pushButtonAlgorithmusSuchenEinstellungen.clicked.connect(mW_pushButtonAlgorithmusSuchenEinstellungen)
    mW.checkBoxAlgorithmusKorrigieren.stateChanged.connect(mW_checkBoxAlgorithmusKorrigieren)
    mW.pushButtonAlgorithmusKorrigierenEinstellungen.clicked.connect(mW_pushButtonAlgorithmusKorrigierenEinstellungen)
    mW.checkBoxAlgorithmusFFK.clicked.connect(mW_checkBoxAlgorithmusFFK)    # statt clicked kann auch toggled verwendet werden stateChanged funktioniert nicht

    def classification():
        print("Hello")

    mW.pushKlassifizieren.clicked.connect(classification)

    ### Flat-Field-Korrektur
    fF.radioButtonGespeicherteBilder.clicked.connect(fF_radioButtonGespeicherteBilder)
    fF.pushButtonHellAdd.clicked.connect(fF_pushButtonHellAdd)
    fF.pushButtonHellDelete.clicked.connect(fF_pushButtonHellDelete)
    fF.pushButtonHellDeleteAll.clicked.connect(fF_pushButtonHellDeleteAll)
    fF.pushButtonDunkelAdd.clicked.connect(fF_pushButtonDunkelAdd)
    fF.pushButtonDunkelDelete.clicked.connect(fF_pushButtonDunkelDelete)
    fF.pushButtonDunkelDeleteAll.clicked.connect(fF_pushButtonDunkelDeleteAll)
    fF.radioButtonNeueBilder.clicked.connect(fF_radioButtonNeueBilder)
    fF.pushButtonGespeicherteBilder.clicked.connect(fF_pushButtonGespeicherteBilder)
    

    
    ### Einstellungen Suchen
    eS.horizontalSliderSchwellwertHot.valueChanged.connect(eS_horizontalSliderSchwellwertHot)
    eS.horizontalSliderSchwellwertDead.valueChanged.connect(eS_horizontalSliderSchwellwertDead)
    eS.horizontalSliderMovingFensterbreite.valueChanged.connect(eS_horizontalSliderMovingFensterbreite)
    eS.horizontalSliderMovingSchwellwert.valueChanged.connect(eS_horizontalSliderMovingSchwellwert)
    eS.horizontalSliderDynamicSchwellwert.valueChanged.connect(eS_horizontalSliderDynamicSchwellwert)
    eS.pushButtonVorschau.clicked.connect(eS_pushButtonVorschau)

    eS.groupBoxSchwellwert.setToolTip("Hinweise für die Einstellung des Schwellwertfilters: \nJe weiter rechts der Schieberegler ist, \ndesto mehr Pixelfehler werden erkannt. \nEmpfohlener Wert ist 0,95 für helle Pixel bzw. 0,05 für dunkle Pixel.")
    eS.groupBoxDynamic.setToolTip("Hinweise für die Einstellung des Dynamic-Check: \nAchtung ein Schwellwert über 0,1 ist Lebensmüde!")
    eS.groupBoxMoving.setToolTip("Hinweise für die Einstellung des Moving-Window: \nGroße Fensterbreiten haben einen hohen Rechenaufwand.\nJe geringer der Schwellwert ist, desto mehr Fehler werden entdeckt. Empfohlene Fensterbreite 7\nEmpfohlener Schwellwert 3,5")

    ### Einstellungen Korrektur
    eK.horizontalSliderNachbarFensterbreite.valueChanged.connect(eK_horizontalSliderNachbarFensterbreite)
    eK.horizontalSliderGradientFensterbreite.valueChanged.connect(eK_horizontalSliderGradientFensterbreite)
    eK.pushButtonVorschau.clicked.connect(eK_pushButtonVorschau)

    ### Fortschritt Fenster
    fortschritt.pushButtonOeffnen.clicked.connect(fortschritt_pushButtonOeffnen)
    
    #### QT UI anzeigen####
    mW.show()

    def KorrekturExportFkt(BPM,F): # Korrigieren
        flagExportFehler = False
        if mW.checkBoxAlgorithmusKorrigieren.isChecked():
            global bildDaten
            aktuelleZeit = str(datetime.now())[:-7].replace(":","-")    # aktuelle Zeit speichern
            Methode=DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["last_korrekturmethode"]
            fortschritt.textEdit.insertPlainText("Bilder korrigiert und gespeichert: ")
            if mW.checkBoxAlgorithmusFFK.isChecked(): #FCC Vorbereiten
                if mW.radioButtonAlgorithmusNachbar.isChecked():
                    Hell_Mittel_Bild=np.uint16(correction.nachbar2(bildDatenHell,BPM,Methode,int(eK.labelNachbarFensterbreite.text())))
                    Dunkel_Mittel_Bild=np.uint16(correction.nachbar2(bildDatenDunkel,BPM,Methode,int(eK.labelNachbarFensterbreite.text())))
                elif mW.radioButtonAlgorithmusGradient.isChecked():
                    Hell_Mittel_Bild=np.uint16(correction.Gradient(bildDatenHell,BPM,Methode,int(eK.labelGradientFensterbreite.text())))
                    Dunkel_Mittel_Bild=np.uint16(correction.Gradient(bildDatenDunkel,BPM,Methode,int(eK.labelGradientFensterbreite.text())))
            #Korrektur Loop
            for i in range(np.shape(bildDaten)[0]):
                cfg.LadebalkenExport=cfg.LadebalkenExport+1
                if cfg.killFlagThreads==True:
                    return -1
                if mW.radioButtonAlgorithmusNachbar.isChecked():
                    GOOD=np.uint16(correction.nachbar2(bildDaten[i],BPM,Methode,int(eK.labelNachbarFensterbreite.text())))
                elif mW.radioButtonAlgorithmusGradient.isChecked():
                    GOOD=np.uint16(correction.Gradient(bildDaten[i],BPM,Methode,int(eK.labelGradientFensterbreite.text())))
                #if mW.radioButtonAlgorithmusNagao():
                if mW.checkBoxAlgorithmusFFK.isChecked():
                    print("FFC Start")
                    GOOD=correction.Flatfield(GOOD, Hell_Mittel_Bild, Dunkel_Mittel_Bild)[0]
                    print("FFC Ende")
                # Export Aufruf______________________________________________________________
                if np.shape(GOOD) == ():   # wenn GOOD eine -1 (Integer) ist
                    openMessageBox(icon=widgets.QMessageBox.Information, text="Die Auflösung der Bad-Pixel-Map und des Bildes sind unterschiedlich",informativeText="Bitte verwenden Sie andere Bilder.",windowTitle="Unterschiedliche Auflösungen",standardButtons=widgets.QMessageBox.Ok,pFunction=msgButtonClick)
                    fortschritt.textEdit.insertPlainText("\nFehler beim Korrigieren.\n")
                    flagExportFehler = True
                else:
                    exP.exportPictures(pPath= mW.lineEditSpeicherort.text(), pImagename= mW.tableWidgetBilddaten.item(i,0).text(), pImage= GOOD, pZeit= aktuelleZeit)
                    fortschritt.textEdit.insertPlainText(str(cfg.LadebalkenExport)+" ")
            fortschritt.textEdit.insertPlainText("\nKorrektur ist abgeschlossen.\n")       
            if flagExportFehler == False:
                fortschritt.textEdit.insertPlainText("Alle Bilder wurden gespeichert.\n")
        fortschritt.textEdit.insertPlainText("Fertig.\n")
        fortschritt.buttonBox.button(widgets.QDialogButtonBox.Ok).setEnabled(True) # Okay Button able
        if mW.checkBoxAlgorithmusSuchen.isChecked():
            fortschritt.textEdit.insertPlainText("Wenn sie die neue Bad-Pixel-Map speichern möchten drücken Sie bitte OK. Ansonsten bitte Cancel oder Abbrechen drücken.")
        cfg.Global_BPM_Dynamik=0
        cfg.Global_BPM_Moving=0
        cfg.Global_BPM_Multi=0 #Alles wieder zurücksetzen.
        cfg.fehlerSammler["aMW"]=0
        cfg.fehlerSammler["MPPC"]=0
        cfg.fehlerSammler["dC"]=0

    def Prozess(): #Hauptprozess nach Start
        global mittelwertBilder
        global BAD_Ges
        vorschauBild = 0
        #Vorschau Live__________
        if (cfg.Ladebalken > 0 and mW.checkBoxAlgorithmusSuchen.isChecked()): 
            vorschauBild = copy.copy(mittelwertBilder) #Bild erstellen
            if np.shape(cfg.Global_BPM_Dynamik) != ():
                vorschauBild=telemetry.markPixelsVirtuell(bpm=cfg.Global_BPM_Dynamik,pBild=vorschauBild,bgr = 2)#Dynamic =rot
            if np.shape(cfg.Global_BPM_Moving) != ():
                vorschauBild=telemetry.markPixelsVirtuell(bpm=cfg.Global_BPM_Moving,pBild=vorschauBild,bgr = 0)#MovingW = blau
            if np.shape(cfg.Global_BPM_Multi) != ():
                vorschauBild=telemetry.markPixelsVirtuell(bpm=cfg.Global_BPM_Multi,pBild=vorschauBild,bgr = 1) #Multi=grün
            #Vorschau anzeigen...
            cv2.imshow("Gefundene Pixelfehler",vorschauBild)
            #cv2.waitKey(1)
            
            exportPath = exP.exportPicturesEasy(pPath=Speichern.dir_path, pImagename="bpmVorschau.png", pImage=vorschauBild)
            pixmap = gui.QPixmap(exportPath)
            """
            fortschritt.label.setPixmap(pixmap)
            fortschritt.label.setScaledContents(True)
            fortschritt.label.resize(pixmap.width(), pixmap.height())
            """

        if cfg.LadebalkenMax != 0:
            fortschritt.progressBar.setValue(int(cfg.Ladebalken/cfg.LadebalkenMax*100))
        else:
            fortschritt.progressBar.setValue(100)
        print("ladebalken = ",cfg.Ladebalken)
   
        #Abfrage Fertig_________
        FertigFlag=False
        cfg.lock.acquire()
        if cfg.Ladebalken==cfg.LadebalkenMax:
            fortschritt.progressBar.setValue(int(100))
            print("Done")
            FertigFlag=True
        cfg.lock.release()

        if FertigFlag:              
            timer.stop()
            if np.shape(cfg.Global_BPM_Dynamik) != ():
                vorschauBild=telemetry.markPixelsVirtuell(bpm=cfg.Global_BPM_Dynamik,pBild=vorschauBild,bgr = 2)#Dynamic =rot
            if np.shape(cfg.Global_BPM_Moving) != ():
                vorschauBild=telemetry.markPixelsVirtuell(bpm=cfg.Global_BPM_Moving,pBild=vorschauBild,bgr = 0)#MovingW = blau
            if np.shape(cfg.Global_BPM_Multi) != ():
                vorschauBild=telemetry.markPixelsVirtuell(bpm=cfg.Global_BPM_Multi,pBild=vorschauBild,bgr = 1) #Multi=grün
            if mW.checkBoxAlgorithmusSuchen.isChecked():
                cv2.imshow("Gefundene Pixelfehler",vorschauBild)    # so kommt beim Mac immer alle Fehler der BPM
            #Zusammenfassen + Speichern oder Laden
            if mW.checkBoxAlgorithmusSuchen.isChecked():
                fortschritt.textEdit.insertPlainText("Pixelfehler-Suche ist abgeschlossen.\n")
                BAD_Ges=detection.Mapping(cfg.Global_BPM_Moving,cfg.Global_BPM_Multi,cfg.Global_BPM_Dynamik)*100 #Digital*100

                # TODO: HERE IS THE BAD PIXELS MAP IN TOTAL

                #BPM Speichern vill auch am Ende.
                # Speichern.BPM_Save(BAD_Ges*150,mW.comboBoxBPMSensor.currentText()) #BPM Speichern    #Nur wenn alles gut war!  und wenn Pixel gesucht wurden.
                Fehlerzahl=cfg.fehlerSammler["aMW"]+cfg.fehlerSammler["MPPC"]+cfg.fehlerSammler["dC"]
                print("BPM enthaelt ",Fehlerzahl," Pixel")
                DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Anz_PixelFehler"]=Fehlerzahl #anzahl an pixeln.
                DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Anz_Bilder"]=DATA["Sensors"][int(mW.comboBoxBPMSensor.currentIndex())]["Anz_Bilder"]+anzahlBilder
                #Abschließend noch Speichern in JSON!
            else: #laden
                # BAD_Ges=Speichern.BPM_Read(mW.comboBoxBPMSensor.currentText())        # Alter Stand, wo es nur eine BPM pro Sensor gibt
                BAD_Ges=Speichern.BPM_Read_Selected(mW.comboBoxBPMBPM.currentText())
                #print(mW.comboBoxBPMBPM.currentText())
                if np.shape(BAD_Ges) ==(): #Wenns noch keine gibt.
                    fortschritt.textEdit.insertPlainText("Es gibt noch keinen Datensatz. Suchen Erforderlich!\n")
                    return
            ID_T=threading.Thread(name="Korrektur",target=KorrekturExportFkt,args=(BAD_Ges,12))
            ID_T.start()
            #exP.exportPictures(pPath= mW.lineEditSpeicherort.text(), pImagename= "Vorschau", pImage= vorschauBild, pZeit= "aktuelleZeit") #Debug Vorschau
            updateBPM()  # Tab 1 updaten
            updateTextBPM()

    timer = core.QTimer()
    timer.timeout.connect(Prozess)


    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()        # für das fbs
    sys.exit(exit_code)