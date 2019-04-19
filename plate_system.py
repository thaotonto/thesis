from darkflow.net.build import TFNet
from collections import Counter
from multiprocessing import Pool
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import time
import os
import re
import argparse
import requests
import json

import detect_characters as DetectCharacters
import detect_plates as DetectPlates
import possible_plate as PossiblePlate

# define constant
BASE_URL = "http://5f7fe4bb.ngrok.io"
URL_CHECK_IN = BASE_URL + "/api/records/check-in"
URL_CHECK_OUT = BASE_URL + "/api/records/check-out"
URL_IMAGE = BASE_URL + "/api/photos/raw/%s"
CONFIDENCE_RATE = 0.3
LICENSE_PLATE_COUNT = 3

# define global varialbles
processing = False
licensePlate = []
displayPlate = False
plateToDisplay = None
last_sent_plate = None

# define the model options for YOLO and run
options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'threshold': 0.3,
    'load': 750
}


def process_image(img_plate):
    list_of_possible_plates = DetectPlates.detect_plates_in_image(img_plate)
    list_of_possible_plates = DetectCharacters.detect_chars_in_plates(list_of_possible_plates)

    if len(list_of_possible_plates) == 0:
        return False
    else:

        list_of_possible_plates.sort(key = lambda possible_plate: len(possible_plate.strChars), reverse = True)

        license_plate_number = ''
        for list_of_possible_plate in list_of_possible_plates:
            if len(list_of_possible_plate.strChars) > 0:
                if re.match(r'\d{5}', list_of_possible_plate.strChars):
                    license_plate_number += list_of_possible_plate.strChars
                elif re.match(r'(?<!\d)\d{4}(?!\d)$', list_of_possible_plate.strChars):
                    license_plate_number += list_of_possible_plate.strChars
                elif re.match(r'\d{2}[A-Z]\d', list_of_possible_plate.strChars):
                    license_plate_number = ''.join((list_of_possible_plate.strChars, license_plate_number))

        if len(license_plate_number) == 0:
            return False
        # end if
    # end if else

    # return true if license plate number is writen in right patten
    if re.match(r'\d{2}[A-Z]\d\d{5}', license_plate_number) or re.match(r'\d{2}[A-Z]\d\d{4}', license_plate_number):
        if license_plate_number == last_sent_plate:
            return False
        else:
            return license_plate_number, img_plate
    else:
        return False
# end funtion


def mycallback(result):
    global processing
    global displayPlate
    global plateToDisplay

    if result[0] != False:
        license_plate_number, imgPlate = result[0][0], result[0][1]

        if addPossiblePlate(license_plate_number):
            displayPlate = True
            plateToDisplay = (license_plate_number, imgPlate)

    processing = False
# end function


# check if plate number read from plate is constant
def addPossiblePlate(license_plate_number):
    global licensePlate

    licensePlate.append(license_plate_number)
    c = Counter(licensePlate)

    if c.most_common(1)[0][1] >= LICENSE_PLATE_COUNT:
        licensePlate = []
        c.clear()

        return True
    else:
        return False
# end function

if __name__ == "__main__":
    ap = argparse.ArgumentParser()                                      # setup argument
    ap.add_argument("-vid", "--video", type=str)                        # argument for load video
    ap.add_argument("-m", "--mode", type=str)
    args = vars(ap.parse_args())

    tfnet = TFNet(options)                                              # setup the model options
    training_result = DetectCharacters.load_data_and_train()            # training OCR

    if training_result == False:                                        # if KNN training was not successful
        print("\nerror: Traning was not successful\n")
    # end if

    pool = Pool()

    capture = cv2.VideoCapture(args["video"])                           # load video
    camera = cv2.VideoCapture(0)                                        # load camera

    colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

    cv2.namedWindow('Plate Image')
    cv2.moveWindow('Plate Image', 20, 20)
    cv2.namedWindow("cam")
    cv2.moveWindow("cam", 20, 500)

    while (capture.isOpened()):
        ret, frame = capture.read()
        if ret:
            results = tfnet.return_predict(frame)                       # plate detection
            for color, result in zip(colors, results):
                if result['confidence'] > CONFIDENCE_RATE:              # detec chars if high confidence
                    if processing == False:
                        processing = True
                        h = result['bottomright']['y'] - result['topleft']['y']
                        w = result['bottomright']['x'] - result['topleft']['x']
                        crop_img = frame[
                            result['topleft']['y']:result['topleft']['y']+h, result['topleft']['x']:result['topleft']['x']+w]
                        resized_crop = cv2.resize(crop_img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
                        async_result = pool.map_async(process_image, (resized_crop,), callback=mycallback)
                        # process_image(resized_crop)

                # draw box on plate
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                frame = cv2.rectangle(frame, tl, br, color, 7)
                text = result['label'] + ', ' + str(result['confidence'])
                frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # resize frame
            resized = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)

            if displayPlate == True:                                 # display plate on screen
                # write plate image for test
                send_time = time.time()
                data = {"plateNumber": plateToDisplay[0]}
                print(data)
                if args["mode"]=="out":
                    r = requests.patch(URL_CHECK_OUT, data=data)
                else:
                    # capture camera
                    ret_cam, frame_cam = camera.read()
                    resized_cam = cv2.resize(frame_cam, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)

                    # prepare data
                    cv2.imwrite("platePhoto" + "-%d.jpg" % send_time, plateToDisplay[1])
                    cv2.imwrite("driverPhoto" + "-%d.jpg" % send_time, resized_cam)
                    multiple_files = [('platePhoto', ("platePhoto" + "-%d.jpg" % send_time,
                                        open("platePhoto" + "-%d.jpg" % send_time, 'rb'), 'image/jpg')),
                                      ('driverPhoto', ("driverPhoto" + "-%d.jpg" % send_time,
                                        open("driverPhoto" + "-%d.jpg" % send_time, 'rb'), 'image/jpg'))]
                    # r = requests.post(URL_CHECK_IN, files=multiple_files, data=data)
                    os.remove("platePhoto" + "-%d.jpg" % send_time)
                    os.remove("driverPhoto" + "-%d.jpg" % send_time)

                # print('server return code: ' + str(r.status_code))
                # if r.status_code == 200:
                #     last_sent_plate = plateToDisplay[0]
                #     if args["mode"]=="out":
                #         parsed = json.loads(r.content)
                #         platePhotoReq = requests.get(URL_IMAGE % parsed['platePhotoId'])
                #         driverPhotoReq = requests.get(URL_IMAGE % parsed['driverPhotoId'])

                #         platePhotoPIL = np.array(Image.open(BytesIO(platePhotoReq.content)).convert('RGB'))
                #         driverPhotoPIL = np.array(Image.open(BytesIO(driverPhotoReq.content)).convert('RGB'))

                #         platePhoto = cv2.cvtColor(platePhotoPIL, cv2.COLOR_RGB2BGR)
                #         driverPhoto = cv2.cvtColor(driverPhotoPIL, cv2.COLOR_RGB2BGR)

                #         cv2.imshow("Plate Image", platePhoto)
                #         cv2.imshow("cam", driverPhoto)

                #     else:
                #         cv2.imshow('Plate Image', plateToDisplay[1])
                #         cv2.imshow("cam", resized_cam)
                cv2.imshow('Plate Image', plateToDisplay[1])
                cv2.imshow("cam", resized_cam)

                # reset
                plateToDisplay = None
                displayPlate = False
            else:
                cv2.imshow('frame', resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            capture.release()
            camera.release()
            cv2.destroyAllWindows()
            pool.close()
            pool.join()
            break
