import importPictures as imP
import os
import detection as dt
import cv2


def loadImages(path):
    image_files = []
    # Import all images! or an image
    if os.path.isdir(path):  # os.path.exists(dirname): # wenn der Pfad überhaupt existiert
        files = os.listdir(path)
        for current_file in files:
            ext = (os.path.splitext(current_file)[1]).lower()
            if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".tif" or ext == ".tiff" or ext == ".his":
                abspath = os.path.abspath(os.path.join(path, current_file))
                image_files.append(abspath)
                rows, cols, anzahlHisBilder, farbtiefe = imP.getAufloesungUndAnzahlUndFarbtiefe(abspath)
    else:
        ext = (os.path.splitext(path)[1]).lower()
        if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".tif" or ext == ".tiff" or ext == ".his":
            image_files.append(path)
    return image_files


if __name__ == '__main__':
    image_path = r"D:\Master EU4M Semester 2\Detektor\Daten-20210518T195907Z-001\Daten\Aufnahmen zur Korrektur Panel Version 2\Serie4\Bildserie4_65kV_20uA_beta.png"
    image_folder = r"D:\Master EU4M Semester 2\Detektor\Daten-20210518T195907Z-001\Daten\Aufnahmen zur Korrektur Panel Version 2\Serie4"

    pictures = loadImages(image_folder)

    data_images = imP.importUIFunction(pictures, pMittelwert=True, pExport=False)
    ResBPM, Counter = dt.advancedMovingWindow(data_images)
    print(ResBPM)
    print(Counter)
    cv2.imwrite("myBPM.png", ResBPM)




