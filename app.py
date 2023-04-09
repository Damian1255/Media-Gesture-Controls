import time
import cv2
import win32api
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from win32con import VK_MEDIA_PLAY_PAUSE, VK_VOLUME_DOWN, VK_VOLUME_UP,VK_MEDIA_NEXT_TRACK, VK_MEDIA_PREV_TRACK, KEYEVENTF_EXTENDEDKEY

# Initialization
model = load_model('mp_hand_gesture')
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

print("\nGuestures Controls: \nStop = Play/Pause \nThumbs Up = Volume Up \nThumbs Down = Volume Down \nPeace = Next \nRock = Previous\n")

while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Predictions
            prediction = model.predict([landmarks], verbose=0)
            classID = np.argmax(prediction)
            confidence = prediction[0][np.argmax(prediction)]

            # Actions
            if classID == 5 and confidence > 0.98:
                win32api.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
                time.sleep(0.5)
            if classID == 2 and confidence > 0.98:
                win32api.keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_EXTENDEDKEY, 0)
                time.sleep(0.25)
            if classID == 3 and confidence > 0.98:
                win32api.keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_EXTENDEDKEY, 0)
                time.sleep(0.25)
            if classID == 1 and confidence > 0.98:
                win32api.keybd_event(VK_MEDIA_NEXT_TRACK, 0, KEYEVENTF_EXTENDEDKEY, 0)
                time.sleep(0.5)
            if classID == 6 and confidence > 0.98:
                win32api.keybd_event(VK_MEDIA_PREV_TRACK, 0, KEYEVENTF_EXTENDEDKEY, 0)
                time.sleep(0.5)
