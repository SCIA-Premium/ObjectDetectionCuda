import cv2
import sys
import os

video_name = sys.argv[1]
video_path = sys.argv[2]

if not os.path.exists('../samples/' + video_name):
    os.mkdir('../samples/' + video_name)

capture = cv2.VideoCapture(video_path)

frameNr = 0

while (True):
    success, frame = capture.read()

    if frameNr % 10 == 0:
        if success:
            cv2.imwrite(f'../samples/{video_name}/{frameNr}.jpg', frame)
        else:
            break

    frameNr = frameNr+1

capture.release()