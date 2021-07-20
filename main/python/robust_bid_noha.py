import copy

import cv2
import numpy as np
from skimage.exposure import rescale_intensity

image_path = r"/Bildserie4_65kV_20uA_beta.png"

F_H = np.array([[-1, 0, -1, 0, -1],
                [0, 0, 0, 0, 0],
                [2, 0, 2, 0, 2],
                [0, 0, 0, 0, 0],
                [-1, 0, -1, 0, -1]])

F_V = np.array([[-1, 0, 2, 0, -1],
                [0, 0, 0, 0, 0],
                [-1, 0, 2, 0, -1],
                [0, 0, 0, 0, 0],
                [-1, 0, 2, 0, -1]])

F_45 = np.array([[-1, 0, -1, 0, 2],
                 [0, 0, 0, 0, 0],
                 [-1, 0, 2, 0, -1],
                 [0, 0, 0, 0, 0],
                 [2, 0, -1, 0, -1]])

F_135 = np.array([[2, 0, -1, 0, -1],
                  [0, 0, 0, 0, 0],
                  [-1, 0, 2, 0, -1],
                  [0, 0, 0, 0, 0],
                  [-1, 0, -1, 0, 2]])


def get_omega(x, y, image, pad):
    pos = np.array([(x - pad, y - pad), (x - pad, y), (x - pad, y + pad), (x, y - pad),
              (x, y + pad), (x + pad, y - pad), (x + pad, y), (x + pad, y + pad)])
    omega = np.array([image[y_i, x_i] for y_i, x_i in pos])
    return omega


def get_i_average_and_second_most(omega):
    i_arr = copy.copy(omega)
    val = [int(a) + int(b) + int(c) for a, b, c in i_arr]
    ind_min = val.index(np.min(val))
    i_arr = np.delete(i_arr, ind_min, 0)
    val = [int(a) + int(b) + int(c) for a, b, c in i_arr]
    ind_max = val.index(np.max(val))
    i_arr = np.delete(i_arr, ind_max, 0)
    val = [int(a) + int(b) + int(c) for a, b, c in i_arr]
    ind_max = val.index(np.max(val))
    ind_min = val.index(np.min(val))
    return np.average(i_arr, axis=0), i_arr[ind_max], i_arr[ind_min]


def check_hot_or_cold(x, y, image, average, bpm):
    if bpm[y, x] != 0:  # if bad pixels
        val = image[y, x]
        if np.all(val > average):
            return 1    # hot
        else:
            return 0    # cold
    else:
        return -1


def get_direction(pix_stat, RH, RV, R45, R135):
    rh_val = np.sum(RH)
    rv_val = np.sum(RV)
    r45_val = np.sum(R45)
    r135_val = np.sum(R135)
    ret = ["RH", "RV", "R45", "R135"]
    val = [rh_val, rv_val, r45_val, r135_val]
    if pix_stat == 1:   # hot pixel
        return ret[val.index(np.max(val))]
    elif pix_stat == 0: # cold
        return ret[val.index(np.min(val))]
    else:   # not in bpm
        return -1


def get_directional_estimate(x, y, image, pad, direction):
    if direction == "RH":
        return np.average([image[y-pad, x], image[y+pad, x]], axis=0)
    elif direction == "RV":
        return np.average([image[y, x-pad], image[y, x+pad]], axis=0)
    elif direction == "R45":
        return np.average([image[y+pad, x-pad], image[y-pad, x+pad]], axis=0)
    elif direction == "R135":
        return np.average([image[y-pad, x-pad], image[y+pad, x+pad]], axis=0)
    else:   # not BPM
        return -1


def get_non_directional_estimate(max_sec, min_sec, pix_stat):
    if pix_stat == 1:   # hot
        return max_sec
    elif pix_stat == 0:     # cold
        return min_sec
    else:   # not bpm
        return -1


def noha_robust_correction(image, bpm, M3, size=5, bpm_hot=None, bpm_cold=None):
    (iH, iW) = image.shape[:2]
    # get response filter
    R_H = cv2.filter2D(img, cv2.CV_8U, F_H)
    # cv2.imshow("R_H", R_H)
    R_V = cv2.filter2D(img, cv2.CV_8U, F_V)
    # cv2.imshow("R_V", R_V)
    R_45 = cv2.filter2D(img, cv2.CV_8U, F_45)
    # cv2.imshow("R_45", R_45)
    R_135 = cv2.filter2D(img, cv2.CV_8U, F_135)
    # cv2.imshow("R_135", R_135)
    pad = (size - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    bpm = cv2.copyMakeBorder(bpm, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    for y in np.arange(iH):
        for x in np.arange(iW):
            if bpm[y, x] != 0:  # if in BPM
                omega = get_omega(x, y, image, pad)
                i_av, sd_max, sd_min = get_i_average_and_second_most(omega)
                pxl_stat = check_hot_or_cold(x, y, image, i_av, bpm)
                px_dir = get_direction(pxl_stat, R_H, R_V, R_45, R_135)
                i_dir = get_directional_estimate(x, y, image, pad, px_dir)
                i_nd = get_non_directional_estimate(sd_max, sd_min, pxl_stat)

                if pxl_stat == 1:  # hot
                    if np.all(i_dir > (1+M3)*i_av):
                        output = i_nd
                    else:
                        output = i_dir
                else:   # cold
                    if np.all(i_dir < (1+M3)*i_av):
                        output = i_nd
                    else:
                        output = i_dir
                image[y, x] = output
    # image = rescale_intensity(image, in_range=(0, 255))
    # image = (image * 255).astype("uint8")
    # cv2.imshow("bpm_pad", 255*bpm)
    return image


def calculate_local_brightness(x, y, image, pad):
    omega = get_omega(x, y, image, pad)
    i_av, mx, mn = get_i_average_and_second_most(omega)
    return image[y, x] - i_av


def check_condition_a(x,y, image, average, M1):
    val = image[y, x]
    if np.all(val > (1+M1) * average):
        # condition A for Hot pixel!
        return 1
    elif np.all(val < (1-M1)*average):
        # condition A for Cold pixel
        return 0
    else:
        return -1


def check_condition_b(x, y, image, pad, M2):
    val = image[y, x]
    dbk_pos = [[(x, y - 1), (x, y + 1)],
               [(x - 1, y), (x + 1, y)],
               [[(x - 1, y - 1), (x - 1, y + 1)],
                [(x + 1, y - 1), (x + 1, y + 1)]]]

    conditions_hot = []
    conditions_cold = []
    dbk = calculate_local_brightness(x, y, image, pad)
    for item in dbk_pos:
        if dbk_pos.index(item) == 2:
            a_1, b_1 = item[0][0]
            a_2, b_2 = item[0][1]
            a_3, b_3 = item[1][0]
            a_4, b_4 = item[1][1]
            dbk_1 = calculate_local_brightness(a_1, b_1, image, pad)
            dbk_2 = calculate_local_brightness(a_2, b_2, image, pad)
            dbk_3 = calculate_local_brightness(a_3, b_3, image, pad)
            dbk_4 = calculate_local_brightness(a_4, b_4, image, pad)
            conditions_hot.append(dbk > M2 * np.min([dbk_1, dbk_2, dbk_3, dbk_4]))
            conditions_cold.append(dbk < M2 * np.max([dbk_1, dbk_2, dbk_3, dbk_4]))
        else:
            a_1, b_1 = item[0]
            a_2, b_2 = item[1]
            dbk_1 = calculate_local_brightness(a_1, b_1, image, pad)
            dbk_2 = calculate_local_brightness(a_2, b_2, image, pad)
            conditions_hot.append(dbk > M2 * np.min([dbk_1, dbk_2]))
            conditions_cold.append(dbk < M2 * np.max([dbk_1, dbk_2]))
    if np.all(conditions_hot):
        return 1
    elif np.all(conditions_cold):
        return 0
    else:
        return -1


def noha_detector(image, size, M1, M2):
    (iH, iW) = image.shape[:2]
    pad = size-1 // 2
    BPM = np.zeros((iH, iW), dtype=np.uint8)
    BPM_hot = np.zeros(np.shape(BPM), dtype=np.uint8)
    BPM_cold = np.zeros(np.shape(BPM), dtype=np.uint8)
    # image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # BPM = cv2.copyMakeBorder(BPM, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # BPM_hot = cv2.copyMakeBorder(BPM, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # BPM_cold = cv2.copyMakeBorder(BPM, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    for y in np.arange(iH):
        for x in np.arange(iW):
            try:
                omega = get_omega(x, y, image, pad)
                i_av, sd_max, sd_min = get_i_average_and_second_most(omega)
                condA = check_condition_a(x, y, image, i_av, M1)
                condB = check_condition_b(x, y, image, pad, M2)
                if condA == 1 and condB == 1:
                    BPM_hot[y, x] = 1
                elif condA == -1 and condB == -1:
                    BPM_cold[y, x] = 1
                else:
                    BPM_hot[y, x] = 0
            except IndexError:
                continue
            # cv2.imshow("bpm_hot_noha", 255*BPM_hot)
            # cv2.imshow("bpm_cold_noha", 255*BPM_cold)
            # cv2.waitKey(1)
    BPM = np.array(np.logical_or(BPM_hot, BPM_cold),dtype=np.uint8)
    return BPM, BPM_hot, BPM_cold


def detect_and_corrected(image, size, M1, M2, M3):
    (iH, iW) = image.shape[:2]
    pad = size-1 // 2
    BPM = np.zeros((iH, iW), dtype=np.uint8)
    BPM_hot = np.zeros(np.shape(BPM), dtype=np.uint8)
    BPM_cold = np.zeros(np.shape(BPM), dtype=np.uint8)
    # get response filter
    R_H = cv2.filter2D(img, cv2.CV_8U, F_H)
    # cv2.imshow("R_H", R_H)
    R_V = cv2.filter2D(img, cv2.CV_8U, F_V)
    # cv2.imshow("R_V", R_V)
    R_45 = cv2.filter2D(img, cv2.CV_8U, F_45)
    # cv2.imshow("R_45", R_45)
    R_135 = cv2.filter2D(img, cv2.CV_8U, F_135)
    # cv2.imshow("R_135", R_135)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # bpm = cv2.copyMakeBorder(bpm, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    for y in np.arange(iH):
        for x in np.arange(iW):
            # search bpm
            try:
                omega = get_omega(x, y, image, pad)
                i_av, sd_max, sd_min = get_i_average_and_second_most(omega)
                condA = check_condition_a(x, y, image, i_av, M1)
                condB = check_condition_b(x, y, image, pad, M2)
                if condA == 1 and condB == 1:
                    BPM_hot[y, x] = 1
                    pxl_stat = 1
                elif condA == -1 and condB == -1:
                    BPM_cold[y, x] = 1
                    pxl_stat = 0
                else:
                    pxl_stat = -1
                px_dir = get_direction(pxl_stat, R_H, R_V, R_45, R_135)
                i_dir = get_directional_estimate(x, y, image, pad, px_dir)
                i_nd = get_non_directional_estimate(sd_max, sd_min, pxl_stat)

                if pxl_stat == 1:  # hot
                    if np.all(i_dir > (1 + M3) * i_av):
                        output = i_nd
                    else:
                        output = i_dir
                    image[y, x] = output
                elif pxl_stat == 0:  # cold
                    if np.all(i_dir < (1 + M3) * i_av):
                        output = i_nd
                    else:
                        output = i_dir
                    image[y, x] = output
                else:
                    continue

            except IndexError:
                continue
            cv2.imshow("bpm_hot_noha", 255*BPM_hot)
            cv2.imshow("bpm_cold_noha", 255*BPM_cold)
            cv2.imshow("corrected", image)
            cv2.waitKey(1)
    return image


if __name__ == '__main__':
    img = cv2.imread("my_im.png")
    # res_bpm = cv2.imread(r"D:\BAD-PIXELS\src\main\python\bpm_cold_noha_m1_1_m2_1.png", cv2.IMREAD_GRAYSCALE)
    res_bpm = cv2.imread("my_bpm.png", cv2.IMREAD_GRAYSCALE)
    toshow = res_bpm*255
    bpm = copy.copy(res_bpm)
    cv2.imshow("orig", img)
    cv2.imshow("bpm", toshow)

    # image = detect_and_corrected(img, 5, 0.1, 1, 1)
    # cv2.imwrite("result_detect_and_corrected.png", image)

    res = noha_robust_correction(img, bpm, 1)
    cv2.imshow("result", res)
    cv2.imwrite("result_robust_noha_bad.png", res)
    cv2.waitKey()
    noha_bpm, noha_hot, noha_cold = noha_detector(img, 5, 1, 1)
    cv2.imshow("bpm", noha_bpm)
    cv2.imwrite("this_bpm_noha_m1_1_m2_1.png", noha_bpm)
    # cv2.imshow("bpm_noha", noha_bpm)
    # cv2.imshow("bpm_hot_noha", noha_hot)
    # cv2.imshow("bpm_cold_noha", noha_cold)

    # cv2.imwrite("bpm_hot_noha_m1_1_m2_10.png", noha_hot)
    # cv2.imwrite("bpm_cold_noha_m1_1_m2_10.png",noha_cold)
    cv2.waitKey()
    print("Finish!")