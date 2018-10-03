#To Run: py -3.6 monsterbot.py
# import the necessary packages
import pytesseract
from PIL import Image
import numpy as np
import argparse
import imutils
import urllib.request as urllib
import discord
import asyncio
import aiohttp
import ntpath
import glob
import cv2
import re

#TODO:
# Add a filter in the templateing which checks overlap/intersections of the best fit images to prevent double detection

#Known issues (these probably appear somewhere in the included sample data):
# - Sometimes stops running after many hours. Never tried to figure out why.
# - Ocean stalker skin can register as rust
# - Can register the same area as multiple different loot types (particularly rust/stalker)
# - Candidum shell came in as 0 instead of 8
# - Nineshark fin (64) showed up as 81

TOKEN = '' #Your discord bot token goes here
discord_channel = '' # Your discord channel goes here

def runClient():
    client = discord.Client()

    @client.event
    async def on_message(message):
        print("Channel ID: " + message.channel.id)

        if(message.channel.id != discord_channel):
            return

        # we do not want the bot to reply to itself
        if message.author == client.user:
            return

        # Pleasantries
        if message.content.startswith('!monsterbot'):
            msg = 'Hello {0.author.mention}! This is <Brutal> Nekonen\'s Sea Monster loot parser. Upload a screenshot of your loot!'.format(message)
            await client.send_message(message.channel, msg)
            return

        # Help Message
        if message.content.startswith('!monsterhelp'):
            msg = "Paste, attach or embed a screenshot of your loot, no commands required! For best results, try to crop closely around the blue boxes surrounding the items. Also, the bot can only recognize one type of loot per screenshot"
            await client.send_message(message.channel, msg)
            return

        # goodbot Message
        if message.content.startswith('!goodbot'):
            msg = "<3 thanks bae <3"
            await client.send_message(message.channel, msg)
            return

        # badbot Message
        if message.content.startswith('!badbot'):
            msg = "Hey, I can't always be perfect :'("
            await client.send_message(message.channel, msg)
            return

        #fault message
        if message.content.startswith('!fault'):
            msg = "It's dissboi's fault"
            await client.send_message(message.channel, msg)
            return

        #log issues
        if message.content.startswith('!reportissue'):
            print (message.content)
            msg = "Thanks for the heads up!"
            await client.send_message(message.channel, msg)
            return

        #commands
        if message.content.startswith('!monstercommands'): 
            msg = " !monsterbot\n !monsterhelp\n !goodbot\n !badbot\n !fault\n !monstercommands \n!reportissue (please provide a detailed description of the problem (eg !monsterbot number 8 on ocean stalker skin recognized as 0)"
            await client.send_message(message.channel, msg)
            return

        #Real bizness
        urls = []
        imageUrls = []
        # Get URLs from embeds
        for embed in message.embeds:
            if 'url' in embed:
                urls.append(embed['url'])
        # Get URLs from attachments
        for attachment in message.attachments:
            if 'url' in attachment:
                urls.append(attachment['url'])
        # Get URLs from message string
        text = message.content
        if (".png" in text.lower()) or (".jpg" in text.lower()) or (".jpeg" in text.lower()):
            words = text.split()
            for word in words:
                if (".png" in word.lower()) or (".jpg" in word.lower()) or (".jpeg" in word.lower()):
                    imageUrls.append(word)

        # Get the URLs that actually point to images (only a little redunadnt)
        for url in urls:
            if (".png" in url.lower()) or (".jpg" in url.lower()) or (".jpeg" in url.lower()):
                imageUrls.append(url)
            else:
                return

        if len(urls) > 0:
            allLoot = []
            for url in urls:

                # Fetch images async and do all the things
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            bytes = await resp.read()
                            
                            image = cv2.imdecode(np.frombuffer(bytes, np.uint8), cv2.IMREAD_COLOR)
                            loot = getLootFromImage(image)
                            if(loot):
                                for l in loot:
                                    allLoot.append(l)

                            # We now should have each separate item as an index in allLoot
                            totalValue = 0
                            mil = 1000000.0
                            outputMessage = "I'm still in development. Sorry if I make mistakes. Please rage at @Nekonen if I fuck up.\n"
                            for loot in allLoot:
                                outputMessage += ("{0}x {1} with value of {2:3.1f} million silver.\n").format(loot.lootCount, loot.type, loot.lootValue/mil)
                                totalValue += loot.lootValue
                            outputMessage += ("Total Value: {0:3.1f} million silver").format(totalValue/mil)

                            await client.send_message(message.channel, outputMessage)

                            if totalValue > (100*mil):
                                bragMessage = ("Hey @everyone, {0.author.mention} just pulled in {1:3.1f} million silver. Git payout carried scrubs").format(message, totalValue/mil)
                                await client.send_message(message.channel, bragMessage)
                return

    @client.event
    async def on_ready():
        print('Logged in as')
        print(client.user.name)
        print(client.user.id)
        print('------')

    client.run(TOKEN)

async def url_to_image(session, url):
  with aiohttp.Timeout(10):
    async with session.get(url) as response:
      assert response.status == 200
      return await response.read()

# Class to parse a loot image
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
        self.type = name
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
        #thresh = [170,166,166] #Some other thresholds I was testing
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
                    im[y,x] = [255,255,255]
                else:
                    im[y,x] = [0,0,0]

        im = cv2.resize(im, (0,0), fx=4.0, fy=4.0)
        im = cv2.medianBlur(im, 3)

        detected = pytesseract.image_to_string(im, config='--psm 8 outputbase nobatch digits')
        detected = re.sub("[^0-9]", "", detected)
        # print ("Detected value:" + detected)
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

def getLootFromImage(image):
    #List to store results
    userLoot = []

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--templates", default='templates', help="Path to template image")
    ap.add_argument("-i", "--images", default='images',
        help="Path to images where template will be matched")
    ap.add_argument("-v", "--visualize",
        help="Flag indicating whether or not to visualize each iteration")
    args = vars(ap.parse_args())

    # ver = pytesseract.get_tesseract_version()
    # print (ver) #Should be 3.05.02 I think

    for templatePath in glob.glob(args["templates"] + "/*.png"):
        # load the image image, convert it to grayscale, and detect edges
        template = cv2.imread(templatePath)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]

        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
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
             
                # draw a bounding box around the detected result and display the image
                end = min(endX + 10, endX)
                detected = image[startY:endY, startX:end]

                # pass detected image into the rest of the program
                loot = DetectedLoot(ntpath.basename(templatePath)[:-4], detected)
                userLoot.append(loot)
    return userLoot


runClient()