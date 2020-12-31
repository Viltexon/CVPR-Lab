import numpy as np
import cv2
from PIL import Image

import time

vid = cv2.VideoCapture('input.avi')
japanify_threshold = 20


# japanify by https://github.com/Lucas-C/dotfiles_and_notes/tree/master/languages/python/img_processing
def japanify(img, threshold):
    width, height = img.size
    img = img.load()  # getting PixelAccess
    for j in range(height):
        contrast = contrastpoints(img, j - 1 if j else 0, width, threshold)     # computing contrast of previous row
        m = 0
        for i in range(width):
            if m < len(contrast) and i >= contrast[m]:
                img[i, j] = (0, 0, 0)   # black
                m += 1
        yield 'ROW_COMPLETE'    # progress tracking


def contrastpoints(img, j, width, threshold):
    contrast = []
    for i in range(width - 3):
        ave1 = sum(img[i + 0, j][:3]) / 3
        ave2 = sum(img[i + 1, j][:3]) / 3
        ave3 = sum(img[i + 2, j][:3]) / 3
        ave4 = sum(img[i + 3, j][:3]) / 3
        if abs(ave2 - ave1) > threshold and abs(ave1 - ave3) > (threshold / 2):
            contrast.append(i)
    return contrast


images_arr = []

start_time = time.time()
while vid.isOpened():
    ret, frame = vid.read()
    if ret:

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        img_height = img.size[1]
        list(japanify(img, japanify_threshold))

        img = img.convert('RGB')
        open_cv_image = np.array(img)

        open_cv_image = open_cv_image[:, :, ::-1].copy()

        images_arr.append(open_cv_image)

    else:
        break

times = round(time.time() - start_time, 0)
print(f"time of encoding: {times // 60} minutes {times % 60} seconds")


width, height = img.size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
fps = 20     # Hm hm
out = cv2.VideoWriter('decoded_2.avi', fourcc, fps, (width, height))
for out_frame in images_arr:
    out.write(out_frame)

out.release()
vid.release()
cv2.destroyAllWindows()
