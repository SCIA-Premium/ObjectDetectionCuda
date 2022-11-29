import cv2
import matplotlib.pyplot as plt
import sys
import json


def object_detection():
    show = False
    if (sys.argv[1] == "--show"):
        show = True
        sys.argv.pop(1)
    image_ref = cv2.imread(sys.argv[1])

    # Parameters
    gaussian_radius = 5
    gaussian_sigma = 0
    opening_radius = 5
    closing_radius = 5
    threshold = 80
    min_pixel_value = 30
    min_box_size = 200
    
    json_outputs = {}

    for i in range(2, len(sys.argv)):
        image_test = cv2.imread(sys.argv[i])

        # Convert to grayscale
        gray_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

        # Compute a Gaussian Filter
        blur_ref = cv2.GaussianBlur(
            gray_ref, (gaussian_radius, gaussian_radius), gaussian_sigma)
        blur_test = cv2.GaussianBlur(
            gray_test, (gaussian_radius, gaussian_radius), gaussian_sigma)

        # Compute the difference between the two images
        diff = cv2.absdiff(blur_ref, blur_test)

        # Perform morphological closing/opening with a disk to remove non-meaningful object
        opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (opening_radius, opening_radius))
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_radius, closing_radius))
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, closing_kernel)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, opening_kernel)

        # Compute the threshold
        thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_TOZERO)[1]

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Remove contours with a max pixel value lower than min_pixel_value
        cnts = [c for c in cnts[0] if cv2.minMaxLoc(
            thresh, mask=cv2.drawContours(thresh.copy(), [c], 0, 255, -1))[1] > min_pixel_value]
        # Remove contours with a size smaller than min_box_size
        cnts = [c for c in cnts if cv2.contourArea(c) > min_box_size]

        # Output the bounding box
        json_outputs[sys.argv[i]] = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            json_outputs[sys.argv[i]].append([x, y, w, h])
            cv2.rectangle(image_test, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the images
        if show:
            plt.subplot(2, 2, 1)
            plt.imshow(image_ref)
            plt.title("Reference image")
            plt.subplot(2, 2, 2)
            plt.imshow(image_test)
            plt.title("Test image")
            plt.subplot(2, 2, 3)
            plt.imshow(diff)
            plt.title("Difference")
            plt.subplot(2, 2, 4)
            plt.imshow(thresh)
            plt.title("Threshold")
            plt.show()
    print(json.dumps(json_outputs, indent=4))

if __name__ == '__main__':
    # Compute the object detection on the two images
    if (len(sys.argv) < 3):
        print(
            "Usage: python main.py --show <image_path_ref> <image_path_test> [image_test...]")
        sys.exit(1)
    object_detection()
