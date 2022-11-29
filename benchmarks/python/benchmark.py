import cv2
import timeit

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def absdiff(image1, image2):
    return cv2.absdiff(image1, image2)

def morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

def threshold(image):
    return cv2.threshold(image, 80, 255, cv2.THRESH_TOZERO)[1]

def find_contours(image):
    return cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def pipeline(image1, image2):
    gray1 = grayscale(image1)
    gray2 = grayscale(image2)
    blur1 = gaussian_blur(gray1)
    blur2 = gaussian_blur(gray2)
    diff = absdiff(blur1, blur2)
    morph = morphology(diff)
    thresh = threshold(morph)
    contours = find_contours(thresh)
    return contours
    

if __name__ == '__main__':

    gaussian_radius = 5
    gaussian_sigma = 0
    opening_radius = 5
    closing_radius = 5
    threshold_value = 80
    min_pixel_value = 30
    min_box_size = 200

    image_path_ref = "../samples/SCIA_Premium/0.jpg"
    image_path_test = "../samples/SCIA_Premium/120.jpg"
    image_ref = cv2.imread(image_path_ref)
    image_test = cv2.imread(image_path_test)

    print("--------------------------------------------------------------------")
    print("Benchmark          Time (ns)                                      ")
    print("--------------------------------------------------------------------")
    
    print("Grayscale :       ", round(1000000 * timeit.timeit(lambda: grayscale(image_ref), number=1000)))

    gray_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

    print("Gaussian Blur :   ", round(1000000 * timeit.timeit(lambda: gaussian_blur(gray_ref), number=1000)))

    # Compute a Gaussian Filter
    blur_ref = cv2.GaussianBlur(gray_ref, (gaussian_radius, gaussian_radius), gaussian_sigma)
    blur_test = cv2.GaussianBlur(gray_test, (gaussian_radius, gaussian_radius), gaussian_sigma)

    print("Absdiff :         ", round(1000000 * timeit.timeit(lambda: absdiff(blur_ref, blur_test), number=1000)))

    # Compute the difference between the two images
    diff = cv2.absdiff(blur_ref, blur_test)

    print("Morphology :      ", round(1000000 * timeit.timeit(lambda: morphology(diff), number=1000)))

    # Perform morphological closing/opening with a disk to remove non-meaningful object
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_radius, opening_radius))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_radius, closing_radius))
        
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, closing_kernel)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, opening_kernel)

    print("Threshold :       ", round(1000000 * timeit.timeit(lambda: threshold(diff), number=1000)))

    # Compute the threshold
    thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_TOZERO)[1]

    print("Find Contours :   ", round(1000000 * timeit.timeit(lambda: find_contours(thresh), number=1000)))

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("Pipeline :        ", round(1000000 * timeit.timeit(lambda: pipeline(image_ref, image_test), number=1000)))