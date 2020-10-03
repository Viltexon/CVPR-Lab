import cv2
import numpy as np
import time
import csv

images = []

img0 = cv2.imread("img/0.jpg", cv2.IMREAD_COLOR)

for i in range(1, 26):  # 0-24
    img = cv2.imread(f'img/{i}.jpg', cv2.IMREAD_COLOR)
    images.append(img)

orb = cv2.ORB_create(nfeatures=5000)    # vary

key0, disc0 = orb.detectAndCompute(img0, None)


csv_columns = ['Matches', 'MeanDist', 'Size', 'Time']
results = []

for img in images:
    start_t = time.time()

    key_orb, disc_orb = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(disc0, disc_orb)

    end_t = time.time()

    matches_data = len(matches)/len(disc_orb)
    mean_data = np.average([x.distance for x in matches])
    size_data = img.shape[0]*img.shape[1]
    time_data = end_t - start_t

    data = {'Matches': matches_data, 'MeanDist': mean_data, 'Size': size_data, 'Time': time_data}

    results.append(data)


csv_file = "csv/results.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns, lineterminator = '\n')
        writer.writeheader()
        writer.writerows(results)
except IOError:
    print("I/O error")
