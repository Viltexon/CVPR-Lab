import numpy as np
import cv2

import time

vid = cv2.VideoCapture('input.avi')

BS = 20  # block size
LDSP_Step = 3  # LDSP Step


def findbias(first, second):

    height = first.shape[0] // BS
    width = first.shape[1] // BS
    bias = np.zeros((height, width, 2))

    bias_algorithm(bias, first, second, height,  width)

    return bias


def get_diamond(S):

    vectors = [[S, 0]]

    for i in range(S):
        vectors.append([vectors[-1][0]-1, vectors[-1][1]+1])
    for i in range(S):
        vectors.append([vectors[-1][0]-1, vectors[-1][1]-1])
    # for i in range(S):
    #     vectors.append([vectors[-1][0]+1, vectors[-1][1]-1])
    # for i in range(S-1):
    #     vectors.append([vectors[-1][0]+1, vectors[-1][1]+1])

    return vectors


def bias_algorithm(bias, first, second, height, width):
    for i in range(height):
        for j in range(width):

            block = first[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS]

            min_diff = mad(block, second[i*BS:(i+1)*BS, j*BS:(j+1)*BS])
            imin = 0
            jmin = 0

            if j==0:    # if first - normal LDSP
                diamond_vectors = get_diamond(LDSP_Step)

            else:   # if not first - check motion vector
                LDSP_S = max(abs(bias[i, j-1, 0]), abs(bias[i, j-1, 1]))
                diamond_vectors = [[bias[i, j-1, 0], bias[i, j-1, 1]],
                                   [LDSP_S, 0], [0, LDSP_S], [-LDSP_S, 0], [0, -LDSP_S]]

            for motion_vector in diamond_vectors:
                if i*BS + motion_vector[0] >= 0 and (i+1)*BS + motion_vector[0] <= second.shape[0]:
                    if j*BS + motion_vector[1] >= 0 and (j+1)*BS + motion_vector[1] <= second.shape[1]:
                        motion_vector = [int(x) for x in motion_vector]
                        dif = mad(block, second[i*BS+ motion_vector[0]:(i+1)*BS+ motion_vector[0],
                                         j*BS+ motion_vector[1]:(j+1)*BS+ motion_vector[1]])
                        if min_diff > dif:
                            min_diff = dif
                            imin = motion_vector[0]
                            jmin = motion_vector[1]

            smol_diamond_vectors = [[1, 0], [0, 1], [-1, 0], [0, -1]]

            center_diff = min_diff
            # for k in range(3):
            while True:

                y = i*BS+ imin
                x = j*BS+ jmin

                for motion_vector in smol_diamond_vectors:
                    if y + motion_vector[0] >= 0 and y + BS + motion_vector[0] <= second.shape[0]:
                        if x + motion_vector[1] >= 0 and x + BS + motion_vector[1] <= second.shape[1]:

                            motion_vector = [int(x) for x in motion_vector]
                            dif = mad(block, second[y + motion_vector[0]:y + BS + motion_vector[0],
                                             x + motion_vector[1]:x + BS + motion_vector[1]])
                            if min_diff > dif:
                                min_diff = dif
                                i_pl = motion_vector[0]
                                j_pl = motion_vector[1]

                if center_diff == min_diff:
                    break
                else:
                    center_diff = min_diff
                    imin = imin + i_pl
                    jmin = jmin + j_pl

            bias[i, j, 0] = imin
            bias[i, j, 1] = jmin


def buildFrame(frame, bias):
    newframe = frame.copy()
    for i in range(bias.shape[0]):
        for j in range(bias.shape[1]):
            block = frame[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS, :]
            newframe[int(i * BS + bias[i][j][0]):int((i + 1) * BS + bias[i][j][0]),
            int(j * BS + bias[i][j][1]):int((j + 1) * BS + bias[i][j][1]), :] = block
    return newframe


# Mean Absolute Difference
def mad(a, b):
    return 1 / a.shape[0] / a.shape[1] * np.sum(np.abs(a - b))


difference = 0
count = 0

images_arr = []

start_time = time.time()
while vid.isOpened():
    ret, frame1 = vid.read()
    if ret:
        if count == 0:
            images_arr.append(frame1)

        count += 1
        ret, frame2 = vid.read()
        if ret:

            first = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            second = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            bias = findbias(second, first)
            newframe = buildFrame(images_arr[-1], bias)
            images_arr.append(newframe)
            p = 0.1

            ret, frame3 = vid.read()
            if ret:
                third = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
                bias = findbias(second, third)

                if count % 60:
                    p = (1 - np.sin(1 / (count % 60) * np.pi / 2)) * p
                f = ((frame1.astype('float') * p) + newframe.copy().astype('float').copy() * (1 - p))
                f = np.clip(f, 0, 254).astype(np.uint8)
                newframe = buildFrame(f, bias)
                images_arr.append(newframe)
            else:
                break

        else:
            images_arr.append(frame1)
            break
    else:
        break

times = round(time.time() - start_time, 0)
print(f"time of encoding: {times // 60} minutes {times % 60} seconds")


output_size = frame1.shape[:2][::-1]
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
fps = 20     # Hm hm
out = cv2.VideoWriter('decoded.avi', fourcc, fps, output_size)
for frame in images_arr:
    out.write(frame)

out.release()
vid.release()
cv2.destroyAllWindows()
