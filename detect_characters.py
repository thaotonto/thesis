import os
import cv2
import numpy as np
import math
import random

import preprocess as Preprocess
import possible_character as PossibleCharacter

# define constant
kNearest = cv2.ml.KNearest_create()

# constants for check_if_possible_character, this checks one possible char only (does not compare to another char)
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

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest.setDefaultK(1)                                                             # set default K to 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object

    return True
# end function


def detect_chars_in_plates(list_of_possible_plates):
    if len(list_of_possible_plates) == 0:          # if list of possible plates is empty
        return list_of_possible_plates             # return
    # end if

    # at this point we can be sure the list of possible plates has at least one plate

    for possible_plate in list_of_possible_plates:
        # preprocess to get grayscale and threshold images
        possible_plate.img_grayscale, possible_plate.img_thresh = Preprocess.preprocess(possible_plate.img_plate)

        # increase size of plate image for easier viewing and char detection
        possible_plate.img_thresh = cv2.resize(
            possible_plate.img_thresh, (0, 0), fx = 1.6, fy = 1.6, interpolation = cv2.INTER_CUBIC)

        # threshold again to eliminate any gray areas
        threshold_value, possible_plate.img_thresh = cv2.threshold(possible_plate.img_thresh,
            0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars
        # (without comparison to other chars yet)
        list_of_possible_chars_in_plate = find_possible_chars_in_plate(possible_plate.img_grayscale, possible_plate.img_thresh)

        # given a list of all possible chars, find groups of matching chars within the plate
        list_of_list_of_matching_chars = find_list_of_groups_of_matching_chars(list_of_possible_chars_in_plate)

        if (len(list_of_list_of_matching_chars) == 0):           # if no groups of matching chars were found in the plate

            possible_plate.strChars = ""
            continue
        # end if

        for i in range(0, len(list_of_list_of_matching_chars)):                 # within each list of matching chars
            # sort chars from left to right
            list_of_list_of_matching_chars[i].sort(key = lambda matching_char: matching_char.centerX)
            # and remove inner overlapping chars
            list_of_list_of_matching_chars[i] = remove_inner_overlapping_chars(list_of_list_of_matching_chars[i])
        # end for

        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        len_of_longest_list_of_chars = 0
        index_of_longest_list_of_chars = 0

        for i in range(0, len(list_of_list_of_matching_chars)):
            if len(list_of_list_of_matching_chars[i]) > len_of_longest_list_of_chars:
                len_of_longest_list_of_chars = len(list_of_list_of_matching_chars[i])
                index_of_longest_list_of_chars = i
            # end if
        # end for

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longest_list_of_matching_chars_in_plate = list_of_list_of_matching_chars[index_of_longest_list_of_chars]

        possible_plate.strChars = recognize_chars_in_plate(possible_plate.img_thresh, longest_list_of_matching_chars_in_plate)

    # end of for
    return list_of_possible_plates
# end function

def find_possible_chars_in_plate(img_grayscale, img_thresh):
    list_of_possible_chars = []                        # return value
    img_thresh_copy = img_thresh.copy()

    # find all contours in plate
    contours, npaHierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possible_char = PossibleCharacter.PossibleCharacter(contour)

        # if contour is a possible char, note this does not compare to other chars (yet) . . .
        if check_if_possible_character(possible_char):
            list_of_possible_chars.append(possible_char)
        # end if
    # end if

    return list_of_possible_chars
# end function


def check_if_possible_character(possible_char):
    if (possible_char.boundingRectArea > MIN_PIXEL_AREA
        and possible_char.boundingRectWidth > MIN_PIXEL_WIDTH
        and possible_char.boundingRectHeight > MIN_PIXEL_HEIGHT
        and MIN_ASPECT_RATIO < possible_char.aspectRatio
        and possible_char.aspectRatio < MAX_ASPECT_RATIO):

        return True
    else:
        return False
    # end if
# end function


def find_list_of_groups_of_matching_chars(list_of_possible_chars):
    # with this function, we start off with all the possible chars in one big list
    # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    # note that chars that are not found to be in a group of matches do not need to be considered further
    list_of_groups_of_matching_chars = []                  # return value

    for possible_char in list_of_possible_chars:
        # find all chars in the big list that match the current char
        list_of_matching_chars = find_list_of_matching_chars(possible_char, list_of_possible_chars)

        # also add the current char to current possible list of matching chars
        list_of_matching_chars.append(possible_char)

        # if current possible list of matching chars is not long enough to constitute a possible plate
        if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue
        # end if

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        list_of_groups_of_matching_chars.append(list_of_matching_chars)

        list_of_possible_chars_with_current_matches_removed = []

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        list_of_possible_chars_with_current_matches_removed = list(set(list_of_possible_chars) - set(list_of_matching_chars))

        recursivelist_of_groups_of_matching_chars = find_list_of_groups_of_matching_chars(list_of_possible_chars_with_current_matches_removed)

        # for each list of matching chars found by recursive call
        for recursivelist_of_matching_chars in recursivelist_of_groups_of_matching_chars:
            list_of_groups_of_matching_chars.append(recursivelist_of_matching_chars)
        # end for

        break       # exit for

    # end for

    return list_of_groups_of_matching_chars
# end function

def find_list_of_matching_chars(possible_char, list_of_chars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    list_of_matching_chars = []                # return value

    for possible_matching_char in list_of_chars:
        # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
        # then we should not include it in the list of matches b/c that would end up double including the current char
        if possible_matching_char == possible_char:
            continue                                # so do not add to list of matches and jump back to top of for loop
        # end if

        # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distance_between_chars(possible_char, possible_matching_char)

        fltAngleBetweenChars = angle_between_chars(possible_char, possible_matching_char)

        fltChangeInArea = (float(abs(possible_matching_char.boundingRectArea - possible_char.boundingRectArea))
                           / float(possible_char.boundingRectArea))

        fltChangeInWidth = (float(abs(possible_matching_char.boundingRectWidth - possible_char.boundingRectWidth))
                            / float(possible_char.boundingRectWidth))
        fltChangeInHeight = (float(abs(possible_matching_char.boundingRectHeight - possible_char.boundingRectHeight))
                             / float(possible_char.boundingRectHeight))

        # check if chars match
        if (fltDistanceBetweenChars < (possible_char.diagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY)
            and fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS
            and fltChangeInArea < MAX_CHANGE_IN_AREA
            and fltChangeInWidth < MAX_CHANGE_IN_WIDTH
            and fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            list_of_matching_chars.append(possible_matching_char)
        # end if
    # end for

    return list_of_matching_chars
# end function


def distance_between_chars(first_char, second_char):
    intX = first_char.centerX - second_char.centerX
    intY = first_char.centerY - second_char.centerY

    return math.hypot(intX, intY)
# end function


def angle_between_chars(first_char, second_char):
    x = float(abs(first_char.centerX - second_char.centerX))
    y = float(abs(first_char.centerY - second_char.centerY))

    fltAngleInRad = math.atan2(y, x)
    fltAngleInDeg = math.degrees(fltAngleInRad)

    return fltAngleInDeg
# end function


# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours,
# but we should only include the char once
def remove_inner_overlapping_chars(list_of_matching_chars):
    list_of_matching_chars_with_inner_char_removed = list(list_of_matching_chars)                # return value

    for current_char in list_of_matching_chars:
        for other_char in list_of_matching_chars:
            if current_char != other_char:        # if current char and other char are not the same char . . .
                # if current char and other char have center points at almost the same location . . .
                if distance_between_chars(current_char, other_char) < (current_char.diagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if current_char.boundingRectArea < other_char.boundingRectArea:         # if current char is smaller than other char
                        # if current char was not already removed on a previous pass . . .
                        if current_char in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(current_char)         # then remove current char
                        # end if
                    else:                                                          # else if other char is smaller than current char
                        # if other char was not already removed on a previous pass . . .
                        if other_char in list_of_matching_chars_with_inner_char_removed:
                            list_of_matching_chars_with_inner_char_removed.remove(other_char)           # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return list_of_matching_chars_with_inner_char_removed
# end function


def recognize_chars_in_plate(img_thresh, list_of_matching_chars):
    strChars = ""                                                                   # the chars in the lic plate

    list_of_matching_chars.sort(key = lambda matching_char: matching_char.centerX)     # sort chars from left to right

    for current_char in list_of_matching_chars:
        pt1 = (current_char.boundingRectX, current_char.boundingRectY)
        pt2 = ((current_char.boundingRectX + current_char.boundingRectWidth),
               (current_char.boundingRectY + current_char.boundingRectHeight))

        # crop char out of threshold image
        imgROI = img_thresh[current_char.boundingRectY : current_char.boundingRectY + current_char.boundingRectHeight,
                           current_char.boundingRectX : current_char.boundingRectX + current_char.boundingRectWidth]


        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # resize image

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)               # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)

        strCurrentChar = str(chr(int(npaResults[0][0])))            # get character from results

        strChars = strChars + strCurrentChar                        # append current char to full string
    # end for

    return strChars
# end function
