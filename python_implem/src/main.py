import cv2
import matplotlib.pyplot as plt
import sys

def object_detection(image_path_ref, image_path_test):
    image_ref = cv2.imread(image_path_ref)
    image_test = cv2.imread(image_path_test)

    # Convert to grayscale
    gray_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

    # Compute a Gaussian Filter
    blur_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)
    blur_test = cv2.GaussianBlur(gray_test, (5, 5), 0)

    # Compute the difference between the two images
    diff = cv2.absdiff(blur_ref, blur_test)

    # Perform morphological closing/opening with a disk to remove non-meaningful object
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    # Compute the threshold
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Keep only the blobs with high peaks
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Output the bouding box
    for c in cnts[0]:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image_test, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the images
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

if __name__ == '__main__':    
    # Compute the object detection on the two images
    object_detection(sys.argv[1], sys.argv[2])