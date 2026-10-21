import numpy as np
import os, re
import cv2, glob

class FVImage:
    pass

class Device:
    def __init__(self, device):
        self.device = device
        self.images = {'UT': self.read_images_UT,
                       'SD': self.read_images_SD,
                       'PKU': self.read_images_PKU}

    def read_device(self, datapath, nSub=None, mode='cae'):
        return self.images[self.device](datapath, nSub, mode)

    def read_images_SD(self, pathData, numSample, mode):
        # read images from SDUMLA-HMT database
        # numSamples - number of subjects selected for train
        #		  > integer 

        # read number of subjects
        subjects = sorted(os.listdir(pathData))
        # select subjects based on partition
        if mode == 'cae':
             subjects = subjects[-numSample:]
        elif mode == 'eval':
             subjects = subjects[:numSample]

        imageset = []
        # imageset.append([])
        imgid = -1
        for subject in range(len(subjects)):
            # sdumla stores each hands in different folders
            # read hands one-by-one
            handpath = pathData + subjects[subject] + '/'
            hands = sorted(os.listdir(handpath))
            for h, hand in enumerate(hands):
                # get all images
                imagepath = handpath + hand + '/'
                images = sorted(os.listdir(imagepath))
                fingers = {'index': 0, 'middle': 1, 'ring': 2}
                fngr = ''
                # loop over all images
                for im, image in enumerate(images):
                    namesplit = re.split('[_ .]', image)
                    if namesplit[-1] != 'bmp':
                        continue
                    finger = namesplit[0]
                    img = FVImage()
                    img.image = cv2.imread(imagepath + image, 0)[:, 50:-50]
                    img.f = fingers[finger] + (3 * h)
                    img.p = int(subjects[subject])
                    img.i = int(namesplit[1])
                    if fngr == finger:
                        imageset[imgid].append(img)
                    else:
                        imgid += 1
                        imageset.append([])
                        imageset[imgid].append(img)
                        fngr = finger

        return imageset

    def read_images_UT(self, pathData, numSample, mode):
        # read images from pathData
        # Parameters:
        #   pathData - path to finger vein images
        #       > string
        #   numSample - number of subjects to be read from
        #       the dataset
        #       > integer: default - 20
        #
        # Created on: 1-7-2019

        # finger image struct list
        fingersPerPerson = []
        # get the folders in pathData
        imageFolders = sorted(os.listdir(pathData))
        if mode == 'cae':
            imageFolders = imageFolders[0:numSample]
        elif mode == 'eval':
            imageFolders = imageFolders[-numSample:]

        # Replace len(imageFolders) with numSamples
        for fldr in range(len(imageFolders)):
            # get the images in the folder of a subject
            imagesInFolder = sorted(os.listdir(pathData + imageFolders[fldr]))
            imagesPerFinger = []
            fingerId = 1
            for img in range(len(imagesInFolder)-1):
                # image path
                imgPath = pathData + '/' + imageFolders[fldr] + '/' + imagesInFolder[img]
                ext = imgPath.split('.')[-1]
                if ext != 'png':
                    continue
                fvImage = FVImage()
                fvImage.image = cv2.imread(imgPath,0)
                fvImage.p = fldr + 1
                fvImage.f = int(imagesInFolder[img].split('_')[1])
                fvImage.i = int(imagesInFolder[img].split('_')[2])
                # store the image belonging to the same finger together
                imagesPerFinger.append(fvImage)
                fingerId += 1
                if fingerId > 4:
                    fingerId = 1
                    # store all the finger images belonging to the same finger together
                    fingersPerPerson.append(imagesPerFinger)
                    imagesPerFinger = []

        return fingersPerPerson

    def read_images_PKU(self, datapath, numSample, mode):
        subjects = sorted(os.listdir(datapath))
        if mode == 'cae':
            subjects = subjects[:numSample]
        else:
            subjects = subjects[-numSample:]

        # loop over subjects
        nimages = 0
        imagesPerSubject = []
        for s, sub in enumerate(subjects):
            # each subject has only one finger for the ease of processing
            # add an entry for the finger in imagesPerSubject array
            imagesPerSubject.append([])
            images = glob.glob(datapath + sub + '/*.bmp')
            # loop over images
            for i, image in enumerate(images):
                imageSet = FVImage()
                imageSet.image = cv2.imread(image, 0)
                imageSet.p = s
                imageSet.f = 1# single finger is captured per subject
                imageSet.i = i 
                imagesPerSubject[s].append(imageSet)
                nimages += 1
        print(nimages)
        return imagesPerSubject

