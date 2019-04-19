import cv2
import numpy as np
import math

class PossibleCharacter:

    def __init__(self, _contour):
        self.contour = _contour
        self.boundingRect = cv2.boundingRect(self.contour)
        [x, y, width, height] = self.boundingRect

        self.boundingRectX = x
        self.boundingRectY = y
        self.boundingRectWidth = width
        self.boundingRectHeight = height

        self.boundingRectArea = self.boundingRectWidth * self.boundingRectHeight
        self.centerX = (self.boundingRectX + self.boundingRectX + self.boundingRectWidth) / 2
        self.centerY = (self.boundingRectY + self.boundingRectY + self.boundingRectHeight) / 2
        self.diagonalSize = math.sqrt((self.boundingRectWidth ** 2) + (self.boundingRectHeight ** 2))
        self.aspectRatio = float(self.boundingRectWidth) / float(self.boundingRectHeight)
    # end constructor

# end class
