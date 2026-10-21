import numpy as np
import numpy.matlib as matlib
from register_images import register_images
from fine_register_images import horizontal_alignment_windows, vertical_alignment, correct_rotation


class Pair:
    pass

def generate_genuine_pairs(imageSet, finealign):
    # GENERATE_GENUINE_PATCH_PAIRS returns an array of all genuine patch pairs
    # in the same set
    #
    #   Returns:
    #       patchPairs - list of all genuine patch pairs containing fields for
    #           reference patch
    #           > patchReference: 2D array(uint8)
    #           > patchObject: 2D array(uint8)
    #           > location: 3D array containing patch pair locations as [row, column]
    #               pairs.

    imagePairs = []
    nImageSets = len(imageSet)

    # loop over all images
    index = 0
    for setIndex in range(nImageSets):
        set = imageSet[setIndex]
        # ignore fingers wilt only one samples
        if len(set) == 1:
            continue
        for imageIndexRef in range(len(set)):
            imageRef = set[imageIndexRef]
            # loop over every other image of the same finger
            for imageIndexObj in range(len(set)):
                if imageIndexObj == imageIndexRef:
                    continue
                imageObj = set[imageIndexObj]
                # register object image
                imageObj = register_images(imageRef, imageObj)
                refpair, objpair = horizontal_alignment_windows([imageRef, imageObj])
                if not finealign:
                    objpair = vertical_alignment([refpair, objpair])

                if finealign:
                    objpair = correct_rotation([refpair, objpair])
                    objpair = vertical_alignment([refpair, objpair])
                    imagepair = [refpair.imageNormalised.astype(np.uint8), 
                                 objpair.imageRegistered.astype(np.uint8), 1]
                else:
                    imagepair = [refpair.imageNormalised.astype(np.uint8), 
                                 objpair.imageRegistered.astype(np.uint8), 1]
                index += 1
                # add image pair to array
                imagePairs.append(imagepair)

        # status update
        print('Making train set. Progress genuine pairs: {0:.2f}'
              .format(setIndex/nImageSets*100))

    return imagePairs

def select_object_images(imageSet, setIndex, nObjectImages=3):
    # SELECT_OBJECT_IMAGES selects a certain amount of imposter object images
    #
    #   Parameters:
    #       imageSet - set of images.
    #           > object
    #       setIndex - index of the set of genuine images that should not be selected
    #           > integer
    #       nObjectImages - number of images to be selected
    #           > integer
    #           > default - 3
    #   Returns:
    #       objectImages - array of selected images
    #           > object array
    #

    objectImages = []
    nSets = len(imageSet)

    # Get all identities except the genuine identity
    available_sets = np.array([i for i in range(len(imageSet)) if i != setIndex and len(imageSet[i]) > 0])

    # Select different impostor identities
    selected_sets = np.random.choice(available_sets, size=nObjectImages, replace=False)

    objectImages = []

    for fIdx in selected_sets:
        # Randomly select one image from this identity
        imIdx = np.random.randint(len(imageSet[fIdx]))

        objectImages.append(imageSet[fIdx][imIdx])

    return objectImages
