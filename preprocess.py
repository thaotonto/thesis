import cv2

# define constant
BLOCK_SIZE = 19
WEIGHTED_MEAN = 9


def preprocess(image):
    '''
        this function return a gray scale image and
        threshold image after increase accuracy by increase contrast
        and reduce noise by blur
    '''

    imgGrayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgHighContrast = increase_contrast(imgGrayscale)

    blur = cv2.GaussianBlur(imgHighContrast, (5, 5), 0)

    imgThreshold = cv2.adaptiveThreshold(
        blur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, BLOCK_SIZE, WEIGHTED_MEAN
    )

    return imgGrayscale, imgThreshold
# end function


def increase_contrast(image):
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, structuringElement)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, structuringElement)

    imgPlusTopHat = cv2.add(image, tophat)
    imgPlusTopHatMinusBlackHat = cv2.subtract(imgPlusTopHat, blackhat)

    return imgPlusTopHatMinusBlackHat
# end function
