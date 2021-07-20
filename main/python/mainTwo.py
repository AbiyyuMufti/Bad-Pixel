import copy
import os
import cv2
import numpy as np

from src.main.python import importPictures as imP

from src.main.python import detection as dt

from src.main.python import telemetry


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


def create_binary_mask_inpainting(bpm, p_bild, bgr=1, erstes=False):  # für die Vorschau
    hoehe, breite = np.shape(bpm)
    mask = np.zeros((hoehe, breite), dtype=np.uint8)
    if np.shape(bpm) == ():  # if(np.shape(pBild[2]) !=np.shape(bpm)):
        print("BPM ist leer")
        return p_bild  # Wenn die BPM noch nicht da ist kommt das Orginal zurück.

    for z in range(hoehe):
        for s in range(breite):
            if (bpm[z, s] != 0):
                cv2.circle(mask, (s, z), bgr, (255, 255, 255), -1)
    return mask


def classify_cluster(bpm, size):
    print("clasify cluster")
    hoehe, breite = np.shape(bpm)
    cluster_bpm = np.zeros((hoehe, breite))
    pad = (size-1 )//2
    r_min = 0 - pad
    r_max = 0 + pad + 1
    for z in range(hoehe):
        for s in range(breite):
            if bpm[z, s] != 0:
                for row in range(r_min, r_max):
                    for col in range(r_min, r_max):
                        if (0 <= (z + row) < hoehe) and (0 <= (s + col) < breite) and ((row != 0) and (col != 0)):
                            if bpm[z + row, s + col] != 0:
                                cluster_bpm[z, s] = 1
                                cluster_bpm[z + row, s + col] = 1
    return cluster_bpm


def classify_lines(bpm, im):
    # TODO: use np.nonzeros in klassifikation
    # dst = cv2.Canny(np.uint8(bpm), 50, 200)
    cdst = cv2.cvtColor(np.uint8(im), cv2.COLOR_GRAY2BGR)
    cdst = np.copy(cdst)

    linesP = cv2.HoughLinesP(cdst, 1, np.pi / 180, 0)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdst)


def classify_bpm(bpm, size, min_length=3):
    hoehe, breite = np.shape(bpm)
    horizontal_line_bpm = np.zeros((hoehe, breite), dtype=np.uint8)
    vertical_line_bpm = np.zeros((hoehe, breite), dtype=np.uint8)
    cluster_bpm = np.zeros((hoehe, breite), dtype=np.uint8)
    pad = (size - 1) // 2
    pad = (size-1 )//2
    r_min = 0 - pad
    r_max = 0 + pad + 1
    for z in range(hoehe):
        for s in range(breite):
            if bpm[z, s] != 0:
                if np.all(bpm[z, s:min_length] != 0):
                    horizontal_line_bpm[z, s] = 1
                if np.all(bpm[z:min_length, s] != 0):
                    vertical_line_bpm[z, s] = 1
                for row in range(r_min, r_max):
                    for col in range(r_min, r_max):
                        if (0 <= (z + row) < hoehe) and (0 <= (s + col) < breite) and ((row != 0) and (col != 0)):
                            if bpm[z + row, s + col] != 0:
                                cluster_bpm[z, s] = 1
                                cluster_bpm[z + row, s + col] = 1

                cv2.imshow("cluster", 255 * cluster_bpm)
                cv2.imshow("hor", 255 * horizontal_line_bpm)
                cv2.imshow("ver", 255 * vertical_line_bpm)
                cv2.waitKey(1)

    horizontal_line_bpm = np.array(horizontal_line_bpm, dtype=np.uint8)
    vertical_line_bpm = np.array(vertical_line_bpm, dtype=np.uint8)
    cluster_bpm = np.array(cluster_bpm, dtype=np.uint8)

    return cluster_bpm, horizontal_line_bpm, vertical_line_bpm


def image_inpainting_telea(image, mask):
    dst = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) # FAST MARCHIN METHODES
    return dst


def image_inpainting_navier_stroke(image, mask):
    dst = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS) # FAST Navier-Stokes, Fluid Dynamics
    return dst


if __name__ == '__main__':
    img = cv2.imread("my_im.png")
    res_bpm = cv2.imread("my_bpm.png", cv2.IMREAD_GRAYSCALE)
    toshow = res_bpm * 255
    bpm = copy.copy(res_bpm)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow("orig", img)

    # TODO: Classification !
    BPM = copy.copy(toshow)
    cluster = classify_cluster(res_bpm, 10)
    cv2.imwrite("myBPM.png", res_bpm)
    cv2.namedWindow("Gefundene Pixelfehler")
    cv2.namedWindow("CLuster", cv2.WINDOW_NORMAL)

    # TODO: Create MASKING
    image_to_show = 0
    image_to_show2 = 0
    image_to_show = create_binary_mask_inpainting(res_bpm, image_to_show, 0)
    image_to_show2 = create_binary_mask_inpainting(cluster, image_to_show2)

    # TODO: Image INPAINTING
    cv2.imshow("Orig", img)
    telea = image_inpainting_telea(img, image_to_show)
    navier = image_inpainting_navier_stroke(img, image_to_show)
    cv2.imshow("Gefundene Pixelfehler", image_to_show)
    cv2.imshow("CLuster", image_to_show2)

    cv2.imshow("telea", telea)
    cv2.imshow("navier", navier)

    cv2.waitKey()

    # image_path = r"../../../with_error_gray.png"
    # cv2.waitKey()
    # # image_folder = r"D:\Master EU4M Semester 2\Detektor\Daten-20210518T195907Z-001\Daten\Aufnahmen zur Korrektur Panel Version 2\Serie4"
    # mittelwertBilder = 0
    # image_to_show = 0
    # image_to_show2 = 0
    # pictures = loadImages(image_path)
    # rows, cols, anzahl, farbtiefe = imP.getAufloesungUndAnzahlUndFarbtiefe(pictures[0])
    # print(rows, cols, anzahl, farbtiefe)
    # data_images = imP.importUIFunction(pictures, pMittelwert=True, pExport=False)
    # mittelwertBilder = imP.importUIFunction(pImportPath=pictures, pMittelwertGesamt=True)
    # orig = copy.copy(mittelwertBilder)
    # image_to_show = copy.copy(mittelwertBilder)
    # image_to_show2 = copy.copy(mittelwertBilder)
    # ResBPM, Counter = dt.advancedMovingWindow(data_images, 17, 3.2)



    #

    #

    #

    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # TODO: Create my own matrix --> compare the algorithm

