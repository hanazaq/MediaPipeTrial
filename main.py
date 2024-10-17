import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True: 
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Corrected line
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # mpDraw.draw_landmarks(img, handLms) #key points
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # connected

    cv2.imshow("Image", img)
    cv2.waitKey(1)
