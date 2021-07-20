import copy
import timeit
import cv2
import numpy as np

# image_path = r"zebra.png"
image_path = r"D:\BAD-PIXELS\Bildserie4_65kV_20uA_beta.png"

ifn = lambda v: v + 3  # index from negative
ifz = lambda v: v - 3  # index from zero
oob = lambda x, m: True if x < 0 or x >= m-1 else False # out of boundary


def is_column_defect(d1, bpm):
    count = 0
    r, c = np.shape(bpm)
    for i in d1:
        x, y = i
        if oob(x, c) or oob(y, r):
            continue
        if bpm[y, x] != 0:
            count = count + 1
            if count > 1:
                return True
    return False


def interpolate_defects(di, bpm, image):
    count = 0
    r, c = np.shape(bpm)
    for i in di[ifn(1):]:
        x, y = i
        if oob(x, c) or oob(y, r):
            continue
        if bpm[y, x] != 0:
            n_defect = ifz(count)
            if n_defect == 3:
                x_n, y_n = di[ifn(n_defect - 2)]
            elif n_defect == 1:
                x_n, y_n = di[ifn(n_defect + 2)]
            else:
                x_n, y_n = di[ifn(-n_defect)]
            # if oob(x_n, c) or oob(y_n, r):
            #     continue
            try:
                image[y, x] = image[y_n, x_n]
            except IndexError:
                print(" xx ")
                continue
            print("change in ", x, ", ", y, " in n_d ", n_defect, " to ", x_n, y_n)
        count = count + 1
    return image


def normalize_vec_gradient(di, image):
    r, c = np.shape(image)
    x_m3, y_m3 = di[ifn(-3)]
    x_m2, y_m2 = di[ifn(-2)]
    x_m1, y_m1 = di[ifn(-1)]
    x_p1, y_p1 = di[ifn(1)]
    x_p2, y_p2 = di[ifn(2)]
    x_p3, y_p3 = di[ifn(3)]
    # ignore if out of boundary!
    if oob(x_m1, c) or oob(y_m1, r) or oob(x_m2, c) or oob(y_m2, r) or oob(x_m3, r) or oob(y_m3, r):
        return image
    if oob(x_p1, c) or oob(y_p1, r) or oob(x_p2, c) or oob(y_p2, r) or oob(x_p3, r) or oob(y_p3, r):
        return image
    deriv_min_one = np.subtract(image[y_m1, x_m1], image[y_m3, x_m3], dtype=int)/2
    deriv_plu_one = np.subtract(image[y_p1, x_p1], image[y_p3, x_p3], dtype=int)/2
    di_min_one = np.clip(np.add(image[y_m2, x_m2], deriv_min_one, dtype=float), 0, 255)
    di_plu_one = np.clip(np.add(image[y_p2, x_p2], deriv_plu_one, dtype=float), 0, 255)
    image[y_m1, x_m1] = di_min_one
    image[y_p1, x_p1] = di_plu_one
    print(di_min_one, di_plu_one)
    # deriv_min_one = int(image[y_m1, x_m1]) - int(image[y_m3, x_m3])
    # deriv_plu_one = int(image[y_p1, x_p1]) - int(image[y_p3, x_p3])
    # di_min_one = int(image[y_m2, x_m2]) + int(deriv_min_one)
    # di_plu_one = int(image[y_p2, x_p2]) + int(deriv_plu_one)
    # image[y_m1, x_m1] = np.uint8(di_min_one)
    # image[y_p1, x_p1] = np.uint8(di_plu_one)
    return image


def normalize_vec_gradient_grad(di, image):
    im_di =  np.array([image[y, x] for x, y in di])
    Gradient = np.gradient(im_di)
    print(Gradient)

def edge_biased_weight(d_list, image, k=1):
    delt = np.zeros(7, dtype=np.uint16)
    xi = np.zeros(7, dtype=np.uint16)
    r, c = np.shape(image)
    I = len(d_list)
    for i in range(I):
        di = d_list[i]
        x_m1, y_m1 = di[ifn(-1)]
        x_p1, y_p1 = di[ifn(1)]
        if oob(x_m1, c) or oob(y_m1, r) or oob(x_p1, c) or oob(y_p1, r):
            continue
        delt[i] = abs(int(image[y_m1, x_m1]) - int(image[y_p1, x_p1]))
    sum_delt = np.sum(delt ** k)
    for i in range(I):
        try:
            xi[i] = (1 - ((delt[i]**k) / sum_delt)) / (I - 1)
        except ValueError:
            xi[i] = (1 - 0 / (I - 1))
    return xi, image


def intensity(d_list, xi, image):
    ints = np.zeros(len(d_list))
    # for e, di in zip(d_list, xi):
    for i in range(len(d_list)):
        di = d_list[i]
        x_m1, y_m1 = di[ifn(-1)]
        x_p1, y_p1 = di[ifn(1)]
        e = xi[i]
        if oob(x_m1, c) or oob(y_m1, r) or oob(x_p1, c) or oob(y_p1, r):
            continue
        ints_cur = e * (int(image[y_m1, x_m1]) + int(image[y_p1, x_p1])) / 2
        ints[i] = np.uint8(ints_cur)
    ints_arr = np.sum(ints)
    gr = []
    for di in d_list:
        for y_i, x_i in di:
            try:
                gr.append(np.array(image[y_i, x_i]))
            except IndexError:
                continue
    return ints_arr, image, np.median(gr)


def show_dir(im, d_list):
    im = copy.copy(im)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    cnt = 0
    for di in d_list:
        for din in di:
            if din[0] < 0 or din[1] < 0:
                continue
            if din[0] > c or din[1] > r:
                continue
            im = cv2.circle(im, din, 1, colors[cnt], -1)
        cnt = cnt + 1
    cv2.imshow("im", im)


def in_a_kernel(image, bpm, x, y, k):
    d1 = np.array(
        [(x + 0, y - 3), (x + 0, y - 2), (x + 0, y - 1), (x, y), (x + 0, y + 1), (x + 0, y + 2), (x + 0, y + 3)])
    d2 = np.array(
        [(x - 3, y + 3), (x - 2, y + 2), (x - 1, y + 1), (x, y), (x + 1, y - 1), (x + 2, y - 2), (x + 3, y - 3)])
    d3 = np.array(
        [(x - 3, y + 0), (x - 2, y + 0), (x - 1, y + 0), (x, y), (x + 1, y + 0), (x + 2, y + 0), (x + 3, y + 0)])
    d4 = np.array(
        [(x - 3, y - 3), (x - 2, y - 2), (x - 1, y - 1), (x, y), (x + 1, y + 1), (x + 2, y + 2), (x + 3, y + 3)])
    r, c = np.shape(image)
    if is_column_defect(d1, bpm):
        dlist = [d2, d3, d4]
    else:
        dlist = [d1, d2, d3, d4]
    # TODO: CHECK AGAIN!
    for di in dlist:
        image = interpolate_defects(di, bpm, image)
    cv2.imshow("interpolate", image)
    # normalize_vec_gradient_grad(dlist, image)
    for di in dlist:
        image = normalize_vec_gradient(di, image)
    cv2.imshow("normalize", image)
    show_dir(image, dlist)
    # #
    xi, image = edge_biased_weight(dlist, image, 2)
    cv2.imshow("biased", image)

    cv2.waitKey(10)
    replacement, image, rp = intensity(dlist, xi, image)
    image[y, x] = replacement if replacement != 0 else rp
    bpm[y, x] = 0
    # cv2.waitKey(1)
    return image


from src.main.python import importPictures as imP

from src.main.python import detection as dt

import os


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
    # pictures = loadImages(image_path)
    # data_images = imP.importUIFunction(pictures, pMittelwert=True, pExport=False)
    # mittelwertBilder = imP.importUIFunction(pImportPath=pictures, pMittelwertGesamt=True)
    # img = copy.copy(mittelwertBilder)
    # image_to_show = copy.copy(mittelwertBilder)
    # ResBPM, Counter = dt.advancedMovingWindow(data_images, 17, 3.2)
    # bpm = ResBPM
    img = cv2.imread(r"D:\BAD-PIXELS\src\main\python\my_im.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, c = np.shape(img)
    # cv2.namedWindow("im", cv2.WINDOW_NORMAL)
    # bpm = np.zeros((r, c), dtype=np.uint8)
    res_bpm = cv2.imread(r"D:\BAD-PIXELS\src\main\python\my_bpm.png", cv2.IMREAD_GRAYSCALE)
    toshow = res_bpm*255
    bpm = copy.copy(res_bpm)
    #bpm = cv2.cvtColor(bpm, cv2.COLOR_BGR2GRAY)
    r, c = np.shape(img)
    cv2.imshow("orig", img)
    cv2.imshow("bpm", toshow)
    for y in range(r):
        for x in range(c):
            if bpm[y, x] != 0:
                in_a_kernel(img, bpm, x, y, 1)
    cv2.imshow("result", img)
    cv2.waitKey()
    print("Finish!")
