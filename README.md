# Bad-Pixel-Korrektur-Programm

Ein plattformunabhängiges Python-Programm mit GUI, welches Bildfehler (Bad-Pixel, Cluster sowie Linienfehler) aus graustufen Bilder bestimmen und korrigieren kann. 

## Getting Started

### Verwendete Bibliotheken
* **NumPy** - [Link](https://github.com/numpy/numpy) (Für die Berechnungen von Matrizen)
* **OpenCV für Python** - [Link](https://pypi.org/project/opencv-python-headless/) (Für den Import, Export und Anzeigen von Bildern)
* **PyQt 5** - [Link](https://pypi.org/project/PyQt5/) (für die grafische Benutzeroberfläche)
* **matplotlib** - [Link](https://matplotlib.org) (Für das Anzeigen von Diagrammen)
* **Fman Build System** - [Link](https://build-system.fman.io) (Für die Erstellung des Installers und der Exe)

### Mit folgenden Befehlen können alle notwendigen Bibliotheken installiert werden
pip install fbs PyQt5==5.9.2

pip install numpy

pip install matplotlib

pip install natsort

Für Windows:
pip install opencv-python 

Für macOS:
pip install opencv-python-headless


### Erstellen einer ausführbaren Datei und eines Installers (platformunabhängig)
Für das Erstellen einer ausführbaren Datei und eines Installers wird das fman-build-system verwendet. Ein einführendes Tutorial ist unter folgender Adresse zu finden: https://github.com/mherrmann/fbs-tutorial
Den Hinweis mit der zu verwendeten Python Version ernst nehmen! 

### GUI bearbeiten
Anstatt das umfangreiche QT Studio zu downloaden, kann auch unter folgender Adresse eine abgespeckte Version runtergeladen werden: https://build-system.fman.io/qt-designer-download

### Lebenstipps
Anscheinend mag der Mac keine prints mit Umlauten, er stürtzt dann einfach ab. (nur in der ausführbaren Datei)

## Versionen

0.1 &nbsp;&nbsp;&nbsp;&nbsp;Erste Version.

## Autoren

* **Julian S.** - [JLS666](https://github.com/JLS666)
* **Andreas B.** - [TrandyAndy](https://github.com/TrandyAndy)

See also the list of [contributors](https://github.com/TrandyAndy/Cor-Count/graphs/contributors) who participated in this project.

