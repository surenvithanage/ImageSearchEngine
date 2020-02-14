#importing required packages
import numpy as np
import cv2
#below library is used to check OpenCV version
import imutils

# Used to extract our 3D HSV color histogram from our images
class ColorDescriptor:
    # only takes a single argument, bins.
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    # image we want to describe
    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        # RGB color space to HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # initializing our list of features to quantify and represent our image
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # Compute a 3D HSV color histogram for different regions of the image
        # instead of the entire image
        # For this will use regions-based histograms rather than global-histograms

        # divide the image into four rectangles/segments
        # top-left , right , bottom-left and right
        segments = [ (0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                     (0, cX, cY, h)]

        # construct an elliptical mask representing the center of the image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # extract a mask for each corner of the image,
            # subtracting the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8" )
            cv2.rectangle(cornerMask, (startX, startY) , (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extracting the color histogram from the image
            # then update the feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extracting a color histogram from the elliptical region
        # then update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the image
        # using the supplied number of bins per channel
        hist = cv2.calcHist( [image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])

        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()

        # return the histogram
        return hist

        
    
