import cv2
import matplotlib.pyplot as plt
import sys
import subprocess
import os
import json

path = os.path.dirname(os.path.realpath(__file__))
images_dir = sys.argv[1]
files = list(os.walk(images_dir))[0][2]
files = [os.path.join(images_dir, f) for f in files]
for file in files:
    print(file)

process = subprocess.Popen([path + "/../cpu_implem/build/cpu_implem"] + files, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = process.communicate()[0]
stderr = process.communicate()[1]

json_outputs = json.loads(output.decode("utf-8"))

for path in json_outputs:
    image = cv2.imread(path)
    for box in json_outputs[path]:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(path + "_out.jpg", image)