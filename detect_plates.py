import numpy as np
import cv2
import math
import random

import preprocess as Preprocess
import possible_character as PossibleCharacter
import detect_characters as DetectCharacters
import possible_plate as PossiblePlate

# define constant
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detect_plates_in_image(image):
    list_of_possible_plates = []

    img_grayscale, img_threshold = Preprocess.preprocess(image)    # preprocess image to get grayscale and threshold images

    # find all possible chars in threshold,
    # finds contours that could be characters (without comparison to other chars yet)
    list_of_possible_characters = find_possible_characters_in_image(img_threshold)

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    list_of_groups_matching_chars = DetectCharacters.find_list_of_groups_of_matching_chars(list_of_possible_characters)
    for group_of_matching_chars in list_of_groups_matching_chars:
        possible_plate = extract_plate(image, group_of_matching_chars)

        if possible_plate.img_plate is not None:
            list_of_possible_plates.append(possible_plate)
        # end if
    # end for

    return list_of_possible_plates
# end function


def find_possible_characters_in_image(img_threshold):
    list_of_possible_characters = []

    img_thresh_copy = img_threshold.copy()

    contours, hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        possible_char = PossibleCharacter.PossibleCharacter(contours[i])

        if DetectCharacters.check_if_possible_character(possible_char):
            list_of_possible_characters.append(possible_char)
        # end if
    # end for

    return list_of_possible_characters
# end function


def extract_plate(img_original, list_of_matching_chars):
    possible_plate = PossiblePlate.PossiblePlate()                              # return value

    # sort chars from left to right based on x position
    list_of_matching_chars.sort(key = lambda matching_char: matching_char.centerX)

    # calculate the center point of the plate
    fltPlateCenterX = (list_of_matching_chars[0].centerX + list_of_matching_chars[len(list_of_matching_chars) - 1].centerX) / 2.0
    fltPlateCenterY = (list_of_matching_chars[0].centerY + list_of_matching_chars[len(list_of_matching_chars) - 1].centerY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((list_of_matching_chars[len(list_of_matching_chars) - 1].boundingRectX
                         + list_of_matching_chars[len(list_of_matching_chars) - 1].boundingRectWidth
                         - list_of_matching_chars[0].boundingRectX)
                        * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matching_char in list_of_matching_chars:
        intTotalOfCharHeights = intTotalOfCharHeights + matching_char.boundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(list_of_matching_chars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    y = list_of_matching_chars[len(list_of_matching_chars) - 1].centerY - list_of_matching_chars[0].centerY
    x = list_of_matching_chars[len(list_of_matching_chars) - 1].centerX - list_of_matching_chars[0].centerX
    fltCorrectionAngleInRad = math.atan2(y, x)
    fltCorrectionAngleInDeg = math.degrees(fltCorrectionAngleInRad)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possible_plate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = img_original.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(img_original, rotationMatrix, (width, height))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possible_plate.img_plate = imgCropped

    return possible_plate
# end function
