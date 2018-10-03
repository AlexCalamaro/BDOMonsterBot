# py -3.6 monsters.py
# import the necessary packages
import pytesseract
from PIL import Image
import numpy as np
import argparse
import imutils
import ntpath
import glob
import cv2
import re


class DetectedLoot:
    lootTable = {
        "neidan" : 100000,
        "goldmont": 100000,
        "goldmont_goblet": 100000000,
        "ocean_stalker": 42800,
        "ocean_stalker_whisker": 100000000, 
        "nineshark" : 53700,
        "nineshark_fin" : 100000000,
        "hekaru" : 82000,
        "hekaru_spike" : 100000000,
        "candidum" : 409200,
        "candidum_steel" : 100000000,
        "rust_tongue" : 100000000,
        "rust" : 52400,
    }

    def __init__(self, name, image):
        self.name = name
        self.image = image
        self.lootCount = self.countLoot()
        self.lootValue = self.calculateLootValue(self.name, self.lootCount)

    # Use PyTesseract to detect the number in the image after cropping image a bit
    def countLoot(self):
        height, width = self.image.shape[:2]

        # Lets just exclude the spensive ones from having a wider snap
        xOffset = 0.15
        if(self.name in ["candidum_steel", "rust_tongue", "hekaru_spike", "nineshark_fin", "ocean_stalker_whisker", "goldmont_goblet"]):
            xOffset = 0.6
        elif (self.name in ["rust", "candidum", "ocean_stalker"]):
            xOffset = 0.33

        im = self.image[int(0.60*height):height, int(width*xOffset):width]

        height, width = im.shape[:2]
        #thresh = [170,166,166]
        #thresh = [162,158,158]
        thresh = [140,140,140]
        delta = 30
        if(self.name in ["rust"]):
            thresh = [170,170,170]
            delta = 8
        for x in range(0, width):
            for y in range(0, height):
                channels = im[y,x]

                if(any(channels < thresh) or rgbDelta(channels) > delta):
                #if(channels[0]<thresh[0] or channels[2]<thresh[2]):
                    im[y,x] = [255,255,255]
                else:
                    im[y,x] = [0,0,0]

        #im = removeSmallArtifacts(im)
        #im = cv2.bitwise_not(im)
        im = cv2.resize(im, (0,0), fx=4.0, fy=4.0)
        im = cv2.medianBlur(im, 3)

        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # _,im = cv2.threshold(im,0,150,cv2.THRESH_BINARY)

        cv2.imshow('img',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
        detected = pytesseract.image_to_string(im, config='--psm 8 outputbase nobatch digits')
        detected = re.sub("[^0-9]", "", detected)
        print ("Detected value:" + detected)
        detected = "".join(detected.split())
        return 0 if detected == '' else int(detected)

    def calculateLootValue(self, lootType, lootCount):
        return self.lootTable[lootType] * lootCount

    def printToScreen(self):
        print ("Item:\t" + str(self.name) + "\r\nCount:\t" + str(self.lootCount) + "\r\nValue:\t" + "{:,}".format(self.lootValue) + "\r\n")

def rgbDelta(pixel):
    mi = min(pixel)
    ma = max(pixel)
    return ma-mi

#Turns white pixels black if they have no adjacent white pixels
def removeSmallArtifacts(image):
    height, width = image.shape[:2]
    for x in range(0, width):
        for y in range(0, height):
            if((x > 0) and (x < width-1) and (y > 0) and (y < height-1)):
                channels = image[y,x]
                white = 0
                if all(channels == white):
                    if(not all(image[y-1,x-1]==white) and not all(image[y-1,x]==white) and not all(image[y-1,x+1]==white) and not all(image[y,x-1]==white) and not all(image[y,x+1]==white) and not all(image[y+1,x-1]==white) and not all(image[y+1,x]==white) and not all(image[y+1,x+1] == white)):
                            continue
                    else:
                        #print("Clearing pixel: " + str(x) + ":" + str(y))
                        image[y,x] = [0,0,0]
            else:
                pass
    return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--templates", default='templates', help="Path to template image")
ap.add_argument("-i", "--images", default='images',
    help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
    help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

ver = pytesseract.get_tesseract_version()
print (ver)


for templatePath in glob.glob(args["templates"] + "/*.png"):
    # load the image image, convert it to grayscale, and detect edges
    print ("Template: " + templatePath)
    template = cv2.imread(templatePath)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    #cv2.imshow("Template", template)

    # loop over the images to find the template in
    for imagePath in glob.glob(args["images"] + "/*.png"):
        #print "Image: " + imagePath
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.5, 25)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break


            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
     
            # check to see if the iteration should be visualized
            if args.get("visualize", False):
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                #cv2.imshow("Visualize", clone)
                #cv2.waitKey(0)
     
            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

     
        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        if(found != None):
            (maxVal, maxLoc, r) = found
            # 6 million is an arbitrary value of goodness. Seems to work.
            if(maxVal > 6000000):

                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

                # print str(startX) + ":" + str(endX)
                # print str(startY) + ":" + str(endY)
             
                # draw a bounding box around the detected result and display the image
                #cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                detected = image[startY:endY, startX:endX+10]
                # cv2.imshow("Image", detected)
                # cv2.waitKey(0)
                loot = DetectedLoot(ntpath.basename(templatePath)[:-4], detected)
                loot.printToScreen()






