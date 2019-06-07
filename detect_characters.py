import os
import cv2
import numpy as np
import math
import random

import preprocess as Preprocess
import possible_character as PossibleCharacter

# define constant
kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 5
MIN_PIXEL_HEIGHT = 40

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 120

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def load_data_and_train():
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest.setDefaultK(1)                                                            

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object

    return True
# end function


def detect_chars_in_plates(list_of_possible_plates):
    if len(list_of_possible_plates) == 0:         
        return list_of_possible_plates             
    
    for possible_plate in list_of_possible_plates:
        # preprocess to get grayscale and threshold images
        possible_plate.img_grayscale, possible_plate.img_thresh = Preprocess.preprocess(possible_plate.img_plate)

        # increase size of plate image for easier viewing and char detection
        possible_plate.img_thresh = cv2.resize(
            possible_plate.img_thresh, (0, 0), fx = 1.6, fy = 1.6, interpolation = cv2.INTER_CUBIC)

        # threshold again to eliminate any gray areas
        threshold_value, possible_plate.img_thresh = cv2.threshold(possible_plate.img_thresh,
            0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        list_of_possible_chars_in_plate = find_possible_chars_in_plate(possible_plate.img_grayscale, possible_plate.img_thresh)

        list_of_list_of_matching_chars = find_list_of_groups_of_matching_chars(list_of_possible_chars_in_plate)

        if (len(list_of_list_of_matching_chars) == 0):          
            possible_plate.strChars = ""
            continue

        for i in range(0, len(list_of_list_of_matching_chars)):                
            # sort chars from left to right
            list_of_list_of_matching_chars[i].sort(key = lambda matching_char: matching_char.centerX)

            list_of_list_of_matching_chars[i] = remove_inner_overlapping_chars(list_of_list_of_matching_chars[i])

        len_of_longest_list_of_chars = 0
        index_of_longest_list_of_chars = 0

        for i in range(0, len(list_of_list_of_matching_chars)):
            if len(list_of_list_of_matching_chars[i]) > len_of_longest_list_of_chars:
                len_of_longest_list_of_chars = len(list_of_list_of_matching_chars[i])
                index_of_longest_list_of_chars = i

        longest_list_of_matching_chars_in_plate = list_of_list_of_matching_chars[index_of_longest_list_of_chars]

        possible_plate.strChars = recognize_chars_in_plate(possible_plate.img_thresh, longest_list_of_matching_chars_in_plate)

    return list_of_possible_plates


def find_possible_chars_in_plate(img_grayscale, img_thresh):
    list_of_possible_chars = []                        # return value
    img_thresh_copy = img_thresh.copy()

    contours, npaHierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possible_char = PossibleCharacter.PossibleCharacter(contour)

        if check_if_possible_character(possible_char):
            list_of_possible_chars.append(possible_char)

    return list_of_possible_chars


def check_if_possible_character(possible_char):
    if (possible_char.boundingRectArea > MIN_PIXEL_AREA
        and possible_char.boundingRectWidth > MIN_PIXEL_WIDTH
        and possible_char.boundingRectHeight > MIN_PIXEL_HEIGHT
        and MIN_ASPECT_RATIO < possible_char.aspectRatio
        and possible_char.aspectRatio < MAX_ASPECT_RATIO):

        return True
    else:
        return False


def find_list_of_groups_of_matching_chars(list_of_possible_chars):
    list_of_groups_of_matching_chars = []                  

    for possible_char in list_of_possible_chars:
        list_of_matching_chars = find_list_of_matching_chars(possible_char, list_of_possible_chars)

        list_of_matching_chars.append(possible_char)

        if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        list_of_groups_of_matching_chars.append(list_of_matching_chars)

        list_of_possible_chars_with_current_matches_removed = []

        list_of_possible_chars_with_current_matches_removed = list(set(list_of_possible_chars) - set(list_of_matching_chars))

        recursivelist_of_groups_of_matching_chars = find_list_of_groups_of_matching_chars(list_of_possible_chars_with_current_matches_removed)

        for recursivelist_of_matching_chars in recursivelist_of_groups_of_matching_chars:
            list_of_groups_of_matching_chars.append(recursivelist_of_matching_chars)

        break      

    return list_of_groups_of_matching_chars


def find_list_of_matching_chars(possible_char, list_of_chars):
    list_of_matching_chars = []              

    for possible_matching_char in list_of_chars:
        if possible_matching_char == possible_char:
            continue                               

        fltDistanceBetweenChars = distance_between_chars(possible_char, possible_matching_char)

        fltAngleBetweenChars = angle_between_chars(possible_char, possible_matching_char)

        fltChangeInArea = (float(abs(possible_matching_char.boundingRectArea - possible_char.boundingRectArea))
                           / float(possible_char.boundingRectArea))

        fltChangeInWidth = (float(abs(possible_matching_char.boundingRectWidth - possible_char.boundingRectWidth))
                            / float(possible_char.boundingRectWidth))
        fltChangeInHeight = (float(abs(possible_matching_char.boundingRectHeight - possible_char.boundingRectHeight))
                             / float(possible_char.boundingRectHeight))

        if (fltDistanceBetweenChars < (possible_char.diagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY)
            and fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS
            and fltChangeInArea < MAX_CHANGE_IN_AREA
            and fltChangeInWidth < MAX_CHANGE_IN_WIDTH
            and fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            list_of_matching_chars.append(possible_matching_char)

    return list_of_matching_chars


def distance_between_chars(first_char, second_char):
    intX = first_char.centerX - second_char.centerX
    intY = first_char.centerY - second_char.centerY

    return math.hypot(intX, intY)


def angle_between_chars(first_char, second_char):
    x = float(abs(first_char.centerX - second_char.centerX))
    y = float(abs(first_char.centerY - second_char.centerY))

    fltAngleInRad = math.atan2(y, x)
    fltAngleInDeg = math.degrees(fltAngleInRad)

    return fltAngleInDeg


def remove_inner_overlapping_chars(list_of_matching_chars):
    list_of_matching_chars_with_inner_char_removed = list(list_of_matching_chars)               

    for current_char in list_of_matching_chars:
        for other_char in list_of_matching_chars:
            if current_char != other_char:        
                if distance_between_chars(current_char, other_char) < (current_char.diagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if current_char.boundingRectArea < other_char.boundingRectArea:         
                        if current_char in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(current_char)         
                    else:                                                         
                        if other_char in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(other_char)          

    return list_of_matching_chars_with_inner_char_removed


def recognize_chars_in_plate(img_thresh, list_of_matching_chars):
    strChars = ""                                                                   

    list_of_matching_chars.sort(key = lambda matching_char: matching_char.centerX)     # sort chars from left to right

    for current_char in list_of_matching_chars:
        pt1 = (current_char.boundingRectX, current_char.boundingRectY)
        pt2 = ((current_char.boundingRectX + current_char.boundingRectWidth),
               (current_char.boundingRectY + current_char.boundingRectHeight))

        # crop char out of threshold image
        imgROI = img_thresh[current_char.boundingRectY : current_char.boundingRectY + current_char.boundingRectHeight,
                           current_char.boundingRectX : current_char.boundingRectX + current_char.boundingRectWidth]


        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # resize image
    
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        

        npaROIResized = np.float32(npaROIResized)              

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)

        strCurrentChar = str(chr(int(npaResults[0][0])))            # get character from results

        strChars = strChars + strCurrentChar                        # append current char to full string

    return strChars
