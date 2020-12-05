import numpy as np
import cv2

import time

file = open('encoded.npy', 'rb')
BS = 20  # block size


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
# frame = np.load(file)
# images_arr.append(frame)
cout = 0
while True:
    try:
        frame = np.load(file)
        if cout == 0:
            images_arr.append(frame)
        cout += 1
        try:  
            bias = np.load(file)
            newframe = buildFrame(images_arr[-1], bias)
            images_arr.append(newframe)
            # f=newframe.copy()
            p = 0.1

            bias = np.load(file)
            if cout % 60:
                p = (1-np.sin(1/(cout % 60)*np.pi/2))*p
            f = ((frame.astype('float')*p) + newframe.copy().astype('float').copy()*(1-p))
            f = np.clip(f, 0, 254).astype(np.uint8)
            newframe = buildFrame(f, bias)
            images_arr.append(newframe)

        except:
            images_arr.append(frame)
    except ValueError:
        break

times = round(time.time() - start_time, 0)
print(f"time of decoding: {times // 60} minutes {times % 60} seconds")


output_size = frame.shape[:2][::-1]
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
fps = 20     # Hm hm
out = cv2.VideoWriter('decoded.avi', fourcc, fps, output_size)
for frame in images_arr:
    out.write(frame)

file.close()
out.release()
cv2.destroyAllWindows()
