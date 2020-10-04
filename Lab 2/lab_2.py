import cv2
import numpy as np
import time
import csv

dataSet = 3
descriptor_n = 2

if dataSet == 1:
    img_path = "img"
    img_positive = 100
    img_negative = 20
    if descriptor_n == 1:   # best 72/70
        distance_p = 0.75
        matches_threshold = 4
        nfeatures_orb = 7000
    if descriptor_n == 2:   # best 71/65
        distance_p = 0.81
        matches_threshold = 1
    csv_dataSet = "stand"

if dataSet == 2:
    img_path = "img2"
    img_positive = 102
    img_negative = 18
    if descriptor_n == 1:   # best 76/89
        distance_p = 0.75
        matches_threshold = 4
        nfeatures_orb = 7000
    if descriptor_n == 2:   # best 71/78
        distance_p = 0.81
        matches_threshold = 1
    csv_dataSet = "CarModel"

if dataSet == 3:
    img_path = "img3"
    img_positive = 94
    img_negative = 20
    if descriptor_n == 1:   # best 70/60
        distance_p = 0.75
        matches_threshold = 7
        nfeatures_orb = 5000
    if descriptor_n == 2:   # best 61/70
        distance_p = 0.72
        matches_threshold = 4
    csv_dataSet = "calculator"


images = []

img0 = cv2.imread(img_path + "/0.jpg", cv2.IMREAD_COLOR)

for i in range(1, img_positive + img_negative + 1):
    img = cv2.imread(img_path + f'/{i}.jpg', cv2.IMREAD_COLOR)
    images.append(img)


if descriptor_n == 1:
    csv_descriptor = "ORB"
    descriptor = cv2.ORB_create(nfeatures=nfeatures_orb)

if descriptor_n == 2:
    csv_descriptor = "AKAZE"
    descriptor = cv2.AKAZE_create()


key0, disc0 = descriptor.detectAndCompute(img0, None)


csv_columns = ['Matches', 'MeanDist', 'Size', 'Time']
results = []

for img in images:
    start_t = time.time()
    matches_data = 0
    mean_data = None

    key_1, disc_1 = descriptor.detectAndCompute(img, None)

    if disc_1 is not None:

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(disc0, disc_1, k=2)

        goodMatches = []
        for m, n in matches:
            if m.distance < distance_p * n.distance:
                goodMatches.append(m)

        end_t = time.time()

        if len(goodMatches):

            sourcePoints = np.float32([key0[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            destinationPoints = np.float32([key_1[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            matchesMask = mask.ravel().tolist()

            matchesFinal = [a for a, b in zip(goodMatches, matchesMask) if b]

            matches_data = len(matchesFinal) / len(goodMatches)

            if len(matchesFinal) > matches_threshold:
                mean_data = np.average([x.distance for x in matchesFinal])
            else:
                mean_data = None

    else:
        end_t = time.time()

    size_data = img.shape[0]*img.shape[1]
    time_data = end_t - start_t

    data = {'Matches': matches_data, 'MeanDist': mean_data, 'Size': size_data, 'Time': time_data}

    results.append(data)
    print(len(results), " - ", data['MeanDist'] is not None)

correct_positive = 0
for index in range(0, img_positive):
    if results[index]['MeanDist'] is not None:
        correct_positive += 1

correct_negative = 0
for index in range(img_positive, img_positive + img_negative):
    if results[index]['MeanDist'] is None:
        correct_negative += 1

print("Correct positive: ", correct_positive/img_positive)
print("Correct negative: ", correct_negative/img_negative)


csv_file = "csv/results_" + csv_descriptor + "_" + csv_dataSet + ".csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns, lineterminator = '\n')
        writer.writeheader()
        writer.writerows(results)
except IOError:
    print("I/O error")
