import cv2 as cv
import numpy as np


class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


if __name__ == '__main__':

    image_path = r"D:\Master EU4M Semester 2\Detektor\Daten-20210518T195907Z-001\Daten\Aufnahmen zur Korrektur Panel Version 2\Serie4\Bildserie4_65kV_20uA_beta.png"
    cv.namedWindow("img", cv.WINDOW_NORMAL)
    cv.namedWindow("inpaint", cv.WINDOW_NORMAL)
    img = cv.imread(image_path)
    if img is None:
        print('Failed to load image file:', image_path)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda: ((255, 255, 255), 255))

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            res = cv.inpaint(img_mark, mark, 3, cv.INPAINT_TELEA)
            cv.imshow('inpaint', res)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()

    print('Done')
    cv.destroyAllWindows()