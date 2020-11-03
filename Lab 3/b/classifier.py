import cv2
import numpy as np
import time

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

###############################################
descriptor_n = 1

if descriptor_n == 1:
    distance_p = 0.75
    MIN_MATCH_COUNT = 5
    nfeatures_orb = 6000
    descriptor = cv2.ORB_create(nfeatures=nfeatures_orb)

if descriptor_n == 2:
    distance_p = 0.75
    MIN_MATCH_COUNT = 2
    # descriptor = cv2.AKAZE_create()
    descriptor = cv2.AKAZE_create(threshold=0.001)


# мы обучаем на всех дескрипторах, или же только на хороших?
# если на всех то BFMatcher как и эталонные изображения нам по сути не нужны
# но и размерность тогда будет уф...
def add_sample(d, matches):
    goodMatches = []
    for m, n in matches:
        if m.distance < distance_p * n.distance:
            goodMatches.append(m)
    if len(goodMatches) < MIN_MATCH_COUNT:
        raise Exception("min_match_count")
    # вырезаем из кадра дескрипторы и добавляем их в семпл
    dest_matches = np.zeros(disc_car.shape)     # pog
    for m in goodMatches:
        dest_matches[m.queryIdx, :] = d[m.trainIdx, :]
    return dest_matches.ravel() / 256


###############################################

start_time = time.time()

img_car = cv2.imread("img/0.jpg")
img_stand = cv2.imread("img2/0.jpg")
img_dosimeter = cv2.imread("img3/0.jpg")

key_car, disc_car = descriptor.detectAndCompute(img_car, None)
key_stand, disc_stand = descriptor.detectAndCompute(img_stand, None)
key_dosimeter, disc_dosimeter = descriptor.detectAndCompute(img_dosimeter, None)

bf = cv2.BFMatcher()

train_data = []
y = []
for i in range(121):
    img = cv2.imread(f"img/{i}.jpg")
    k, d = descriptor.detectAndCompute(img, None)
    try:
        matches = bf.knnMatch(disc_car, d, k=2)
        data = add_sample(d, matches)
        train_data.append(data)
        if i <= 102:
            y.append(1)
        else:
            y.append(0)
    except:
        print("min_match_count")


for i in range(121):
    img = cv2.imread(f"img2/{i}.jpg")
    k, d = descriptor.detectAndCompute(img, None)
    try:
        matches = bf.knnMatch(disc_stand, d, k=2)
        data = add_sample(d, matches)
        train_data.append(data)
        if i <= 100:
            y.append(2)
        else:
            y.append(0)
    except:
        print("min_match_count")


for i in range(121):
    img = cv2.imread(f"img3/{i}.jpg")
    k, d = descriptor.detectAndCompute(img, None)
    try:
        matches = bf.knnMatch(disc_dosimeter, d, k=2)
        data = add_sample(d, matches)
        train_data.append(data)
        if i <= 94:
            y.append(3)
        else:
            y.append(0)
    except:
        print("min_match_count")


train_data = np.array(train_data)
y = np.array(y)

# X_train, X_test, Y_train, Y_test = train_test_split(train_data, y, random_state=0, test_size=0.2)

print("Train shape: ")
# print(X_train.shape)
print(train_data.shape)
# print("Test shape: ")
# print(X_test.shape)


clf = AdaBoostClassifier(n_estimators=400)

# clf.fit(X_train, Y_train)
scores = cross_val_score(clf, train_data, y, cv=5)
print(scores.mean())

times = round(time.time() - start_time, 0)
print(f"time of learning: {times // 60} minutes {times % 60} seconds")


########################
