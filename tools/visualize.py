import json
import matplotlib.pyplot as plt
import sys
import cv2

def visualize_bounding_boxes():
    data = sys.stdin.read()
    data = json.loads(data)
    #frameSize = (500, 500)
    #out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, frameSize)
    for path in data:
        image = cv2.imread(path)
        for box in data[path]:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.show()
        #out.write(image)
    #out.release()
        
if __name__ == "__main__":
    visualize_bounding_boxes()