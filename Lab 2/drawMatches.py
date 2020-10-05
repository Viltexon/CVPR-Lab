import numpy as np
import cv2
import time


img0 = cv2.imread("img/0.jpg", cv2.IMREAD_COLOR)
img1 = cv2.imread("img/50.jpg", cv2.IMREAD_COLOR)

nfeatures_orb = 7000
distance_p = 0.75
matches_threshold = 4

descriptor = cv2.ORB_create(nfeatures=nfeatures_orb)
# descriptor = cv2.AKAZE_create()

key0, disc0 = descriptor.detectAndCompute(img0, None)

start_t = time.time()
matches_data = 0
mean_data = None

key_1, disc_1 = descriptor.detectAndCompute(img1, None)

if disc_1 is not None:

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(disc0, disc_1, k=2)

    goodMatches = []
    for m, n in matches:
        if m.distance < distance_p * n.distance:
            goodMatches.append(m)

    if len(goodMatches):

        sourcePoints = np.float32([key0[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        destinationPoints = np.float32([key_1[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        matchesMask = mask.ravel().tolist()

        matchesFinal = [a for a, b in zip(goodMatches, matchesMask) if b]

        matches_data = len(matchesFinal) / len(goodMatches)

        if len(matchesFinal) > matches_threshold:
            mean_data = np.average([x.distance for x in matchesFinal])

            h = img0.shape[0]
            w = img0.shape[1]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img1 = cv2.polylines(img1, [np.int32(dst)], True, (255, 0, 0), 2, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=matchesMask,
                               flags=2)

            img3 = cv2.drawMatches(img0, key0, img1, key_1, goodMatches, None, **draw_params)

            scale_percent = 40  # percent of original size
            width = int(img3.shape[1] * scale_percent / 100)
            height = int(img3.shape[0] * scale_percent / 100)
            dim = (width, height)

            imS = cv2.resize(img3, dim)

            cv2.imshow("Matching result", imS)

        else:
            mean_data = None


end_t = time.time()
size_data = img1.shape[0]*img1.shape[1]
time_data = end_t - start_t

data = {'Matches': matches_data, 'MeanDist': mean_data, 'Size': size_data, 'Time': time_data}

print(data)
print(data['MeanDist'] is not None)

cv2.waitKey(0)
cv2.destroyAllWindows()
