import cv2
import numpy as np
import time

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

###############################################
descriptor_n = 1

if descriptor_n == 1:
    # distance_p = 0.75
    # MIN_MATCH_COUNT = 5
    nfeatures = 500
    pog = 32
    descriptor = cv2.ORB_create(nfeatures=nfeatures)

if descriptor_n == 2:
    # distance_p = 0.75
    # MIN_MATCH_COUNT = 2
    nfeatures = 3000    # why?
    pog = 61
    descriptor = cv2.AKAZE_create(threshold=0.001)


# мы обучаем на всех дескрипторах, или же только на хороших?
# если на всех то BFMatcher как и эталонные изображения нам по сути не нужны
# но и размерность тогда будет уф...
# def add_sample(d, matches):
#     goodMatches = []
#     for m, n in matches:
#         if m.distance < distance_p * n.distance:
#             goodMatches.append(m)
#     if len(goodMatches) < MIN_MATCH_COUNT:
#         raise Exception("min_match_count")
#     # вырезаем из кадра дескрипторы и добавляем их в семпл
#     dest_matches = np.zeros(disc_car.shape)     # pog
#     for m in goodMatches:
#         dest_matches[m.queryIdx, :] = d[m.trainIdx, :]
#     return dest_matches.ravel() / 256


###############################################

start_time = time.time()

# img_car = cv2.imread("img/0.jpg")
# img_stand = cv2.imread("img2/0.jpg")
# img_dosimeter = cv2.imread("img3/0.jpg")
#
# key_car, disc_car = descriptor.detectAndCompute(img_car, None)
# key_stand, disc_stand = descriptor.detectAndCompute(img_stand, None)
# key_dosimeter, disc_dosimeter = descriptor.detectAndCompute(img_dosimeter, None)
#
# bf = cv2.BFMatcher()

train_data = []
y = []
for i in range(121):
    img = cv2.imread(f"img/{i}.jpg")
    k, d = descriptor.detectAndCompute(img, None)
    try:
        # matches = bf.knnMatch(disc_car, d, k=2)
        # data = add_sample(d, matches)
        # train_data.append(data)
        dest_matches = np.zeros((nfeatures, pog))
        for j in range(len(d)):     # smaller
            dest_matches[j, :] = d[j, :]
        train_data.append(dest_matches.ravel() / 256)
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
        # matches = bf.knnMatch(disc_stand, d, k=2)
        # data = add_sample(d, matches)
        # train_data.append(data)
        dest_matches = np.zeros((nfeatures, pog))
        for j in range(len(d)):     # smaller
            dest_matches[j, :] = d[j, :]
        train_data.append(dest_matches.ravel() / 256)
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
        # matches = bf.knnMatch(disc_dosimeter, d, k=2)
        # data = add_sample(d, matches)
        # train_data.append(data)
        dest_matches = np.zeros((nfeatures, pog))
        for j in range(len(d)):     # smaller
            dest_matches[j, :] = d[j, :]
        train_data.append(dest_matches.ravel() / 256)
        if i <= 94:
            y.append(3)
        else:
            y.append(0)
    except:
        print("min_match_count")


train_data = np.array(train_data)
y = np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, y, random_state=0, test_size=0.2)

print("Train shape: ")
print(X_train.shape)
# print(train_data.shape)
print("Test shape: ")
print(X_test.shape)


clf = AdaBoostClassifier(n_estimators=400)

clf.fit(X_train, Y_train)
# scores = cross_val_score(clf, train_data, y, cv=5)
# print(scores.mean())

times = round(time.time() - start_time, 0)
print(f"time of learning: {times // 60} minutes {times % 60} seconds")


########################

t1 = 0
t2 = 0
t3 = 0
t4 = 0
a1 = 0
a2 = 0
a3 = 0
a4 = 0
times = 0
for i in range(X_test.shape[0]):
    start_time = time.time()
    yp = clf.predict(np.expand_dims(X_test[i], axis=0))
    times += (time.time() - start_time)

    yt = Y_test[i]

    # TODO FP & FN
    if yp == yt == 1:
        t1 += 1
    elif yp == yt == 2:
        t2 += 1
    elif yp == yt == 3:
        t3 += 1
    elif yp == yt == 0:
        t4 += 1

    if yp == 1:
        a1 += 1
    elif yp == 2:
        a2 += 1
    elif yp == 3:
        a3 += 1
    elif yp == 0:
        a4 += 1


sY = [0, 0, 0, 0]
sY[0] = np.count_nonzero(Y_test == 1)
sY[1] = np.count_nonzero(Y_test == 2)
sY[2] = np.count_nonzero(Y_test == 3)
sY[3] = np.count_nonzero(Y_test == 0)
print(f"detected car:\t\t{t1/sY[0]}")
print(f"detected stand:\t\t{t2/sY[1]}")
print(f"detected dosimeter:\t{t3/sY[2]}")
print(f"nothing detected:\t{t4/sY[3]}\n")

print(f"false positive car:\t{(a1 - t1)/a1}")
print(f"false positive stand:\t{(a2 - t2)/a2}")
print(f"false positive dosimeter:{(a3 - t3)/a3}")
print(f"false negative:\t\t{(a4 - t4)/a4}\n")   # false positive nothing

print(f"mean time:\t\t{times/X_test.shape[0]}s")
