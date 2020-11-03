import numpy as np
import cv2

import time

file = open('encoded.npy', 'rb')
BS = 8  # block size


def buildFrame(frame, bias):
    newframe = frame.copy()
    for i in range(bias.shape[0]):
        for j in range(bias.shape[1]):
            block = frame[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS, :]
            newframe[int(i * BS + bias[i][j][0]):int((i + 1) * BS + bias[i][j][0]),
            int(j * BS + bias[i][j][1]):int((j + 1) * BS + bias[i][j][1]), :] = block
    return newframe


images_arr = []

start_time = time.time()
while True:
    try:
        frame = np.load(file)
        try:
            images_arr.append(frame)
            bias = np.load(file)
            newframe = buildFrame(frame, bias)
            images_arr.append(newframe)

        except ValueError:
            images_arr.append(frame)
            break
    except ValueError:
        break

times = round(time.time() - start_time, 0)
print(f"time of decoding: {times // 60} minutes {times % 60} seconds")


output_size = frame.shape[:2][::-1]
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
fps = 25     # Hm hm
out = cv2.VideoWriter('decoded.avi', fourcc, fps, output_size)
for frame in images_arr:
    out.write(frame)

file.close()
out.release()
cv2.destroyAllWindows()
