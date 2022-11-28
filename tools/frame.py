import cv2
import sys
import os

video_name = sys.argv[1]
video_path = sys.argv[2]
number_of_frames = int(sys.argv[3])

if not os.path.exists('../samples/' + video_name):
    os.mkdir('../samples/' + video_name)

for file in os.listdir('../samples/' + video_name):
    os.remove('../samples/' + video_name + '/' + file)

capture = cv2.VideoCapture(video_path)

frameNr = 0

while (True):
    success, frame = capture.read()

    if frameNr % number_of_frames == 0:
        if success:
            cv2.imwrite(f'../samples/{video_name}/{frameNr}.jpg', frame)
        else:
            break

    frameNr = frameNr+1

capture.release()