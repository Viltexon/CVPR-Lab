import cv2
import numpy as np
import time
import csv

images = []

img0 = cv2.imread("img/0.jpg", cv2.IMREAD_COLOR)

for i in range(1, 101):  # 0-99
    img = cv2.imread(f'img/{i}.jpg', cv2.IMREAD_COLOR)
    images.append(img)

orb = cv2.ORB_create(nfeatures=5000)    # vary

key0, disc0 = orb.detectAndCompute(img0, None)


csv_columns = ['Matches', 'MeanDist', 'Size', 'Time']
results = []

for img in images:
    start_t = time.time()

    key_orb, disc_orb = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(disc0, disc_orb, k=2)

    goodMatches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # vary
            goodMatches.append(m)

    end_t = time.time()

    if len(goodMatches) > 0:

        sourcePoints = np.float32([key0[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        destinationPoints = np.float32([key_orb[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        matchesMask = mask.ravel().tolist()

        matchesFinal = []
        for i, match in zip(matchesMask, goodMatches):
            if i:
                matchesFinal.append(match)

        matches_data = len(matchesFinal) / len(goodMatches)

        if len(matchesFinal):
            mean_data = np.average([x.distance for x in matchesFinal])
        else:
            mean_data = None

    else:

        matches_data = 0
        mean_data = None

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
