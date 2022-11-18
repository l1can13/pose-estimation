import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # For drawing keypoints
points = mpPose.PoseLandmark  # Landmarks
path = "dataset/images"  # enter dataset path
data = []
for p in points:
    x = str(p)[13:]
    data.append(x + "_x")
    data.append(x + "_y")
    data.append(x + "_z")
    data.append(x + "_vis")
data = pd.DataFrame(columns=data)  # Empty dataset

count = 0

for img in os.listdir(path):
    temp = []
    img = cv2.imread(path + "/" + img)
    imageWidth, imageHeight = img.shape[:2]
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blackie = np.zeros(img.shape)  # Blank image
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image
        mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # draw landmarks on blackie
        landmarks = results.pose_landmarks.landmark

        for i, j in zip(points, landmarks):
            temp = temp + [j.x, j.y, j.z, j.visibility]

        data.loc[count] = temp
        count += 1

    cv2.imshow("Image", img)
    cv2.imshow("blackie", blackie)
    cv2.waitKey(3000)

data.to_csv("dataset3.csv")  # save the data as a csv file

data = pd.read_csv("dataset3.csv")
X, Y = data.iloc[:, :132], data['target']
model = SVC(kernel='poly')
model.fit(X, Y)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
path = "/test images/001.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(imgRGB)
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    for j in landmarks:
        temp = temp + [j.x, j.y, j.z, j.visibility]
    y = model.predict([temp])
    if y == 0:
        asan = "plank"
    else:
        asan = "goddess"
    print(asan)
    cv2.putText(img, asan, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    cv2.imshow("image", img)
