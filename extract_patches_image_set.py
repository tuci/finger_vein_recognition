import numpy as np
import matplotlib.pyplot as plt
from process_image_set import compute_properties_image
import cv2

class Patches:
    pass

def extract_patches_image_set(imageSet, patchSize, slide=None):
    # EXTRACT_PATCHES_IMAGE_SET adds patches to normalised images
    # Patches do not overlap
    # Parameters:
    #   imageSet - set of images. Objects represent a set of images containing the
    #       following fields
    #       > imageNormalised - 2D array(uint8)
    #       > maskNormalised - 2D array(logical)
    #   patchSize - size of patches to be taken
    #       > integer
    #
    #   Returns:
    #       imageSetPatches - following fields are added to imageSet
    #           > patches - object containing the following fields
    #               > patch: 2D array(unit8)
    #               > location: 1D array
    #                   > row: integer
    #                   > column: integer
    # Created on: 3-7-2019

    # field is added, image set is copied
    imageSetPatches = imageSet

    if slide == None:
        slide = patchSize

    # loop over all images
    nImageSet = len(imageSet)
    for s in range(nImageSet):
        nImages = len(imageSet[s])
        for i in range(nImages):
            image = imageSet[s][i]
            imagePatches = extract_patches_image(image, patchSize, slide)
            # add patches filed to image set
            imageSetPatches[s][i].patches = imagePatches.patches

        # print status
        print('Extracting image patches. Progress: {0:.2f}'.format(s/nImageSet*100))

    # print final status
    print('Extracting image patches done ...')
    return imageSetPatches

def extract_patches_image(imageSet, patchSize, slide):
    # field is added, imageSet is copied
    imagePatches = imageSet

    # take image and mask
    image = imageSet.imageNormalised
    mask = imageSet.maskNormalised

    # define patch object
    patchObj = Patches()

    # define patch and location array
    patches = []
    locations = []

    # loop over all non-overlapping patches
    [mRow, nColumn] = image.shape

    h, w = patchSize
    numpatches = 0
    for rowBottom in range(h,mRow, slide[0]):
        rowTop = rowBottom - h
        for columnRight in range(w,nColumn, slide[1]):
            columnLeft = columnRight - w
            # check mask
            onMask = np.all(mask[rowTop:rowBottom, columnLeft:columnRight])
            if ~onMask:
                continue
            # add patch and its location
            patch = image[rowTop:rowBottom,columnLeft:columnRight]
            row = rowTop
            column = columnLeft
            patches.append(patch)
            locations.append([row, column])
    patchObj.patches = patches
    patchObj.locations = locations
    imagePatches.patches = patchObj
    return imagePatches

def extract_borderpatches_image(imageSet, patchSize):
    # field is added, imageSet is copied
    imagePatches = imageSet

    # take image and mask
    image = imageSet.imageNormalised
    mask = imageSet.maskBorder

    # define patch object
    patchObj = Patches()

    # define patch and location array
    patches = []
    locations = []

    # loop over all non-overlapping patches
    [mRow, nColumn] = image.shape

    h, w = patchSize
    numpatches = 0

    for rowBottom in range(h,mRow, 20):
        rowTop = rowBottom - h
        for columnRight in range(w,nColumn, 20):
            columnLeft = columnRight - w
            # check mask
            onMask = np.all(mask[rowTop:rowBottom, columnLeft:columnRight])
            if ~onMask:
                continue
            # add patch and its location
            patch = image[rowTop:rowBottom,columnLeft:columnRight]
            row = rowTop
            column = columnLeft
            patches.append(patch)
            locations.append([row, column])
    patchObj.patches = patches
    patchObj.locations = locations
    imagePatches.patches = patchObj

    return imagePatches

def extract_patches_image_set_noReg(imageSet, patchSize):
    # EXTRACT_PATCHES_IMAGE_SET adds patches to normalised images
    # Patches do not overlap
    # Parameters:
    #   imageSet - set of images. Objects represent a set of images containing the
    #       following fields
    #       > imageNormalised - 2D array(uint8)
    #       > maskNormalised - 2D array(logical)
    #   patchSize - size of patches to be taken
    #       > integer
    #
    #   Returns:
    #       imageSetPatches - following fields are added to imageSet
    #           > patches - object containing the following fields
    #               > patch: 2D array(unit8)
    #               > location: 1D array
    #                   > row: integer
    #                   > column: integer
    # Created on: 3-7-2019

    # field is added, image set is copied
    imageSetPatches = imageSet

    # loop over all images
    nImageSet = len(imageSet)
    for s in range(nImageSet):
        nImages = len(imageSet[s])
        for i in range(nImages):
            image = imageSet[s][i]
            imagePatches = extract_patches_image_noReg(image, patchSize)
            # add patches filed to image set
            imageSetPatches[s][i].patches = imagePatches.patches

        # print status
        print('Extracting image patches. Progress: {0:.2f}'.format(s/nImageSet*100))

    # print final status
    print('Extracting image patches done ...')
    return imageSetPatches

def extract_patches_image_noReg(imageSet, patchSize):
    # field is added, imageSet is copied
    imagePatches = imageSet

    # take image and mask
    image = imageSet.image
    mask = imageSet.mask

    # define patch object
    patchObj = Patches()

    # define patch and location array
    patches = []
    locations = []

    # loop over all non-overlapping patches
    [mRow, nColumn] = image.shape

    for rowBottom in range(patchSize,mRow,patchSize):
        rowTop = rowBottom - patchSize
        for columnRight in range(patchSize,nColumn,patchSize):
            columnLeft = columnRight - patchSize
            # check mask
            onMask = np.all(mask[rowTop:rowBottom, columnLeft:columnRight])
            if ~onMask:
                continue
            # add patch and its location
            patch = image[rowTop:rowBottom,columnLeft:columnRight]
            row = rowTop
            column = columnLeft
            patches.append(patch)
            locations.append([row, column])
    patchObj.patches = patches
    patchObj.locations = locations

    imagePatches.patches = patchObj
    return imagePatches

def get_batches(imageset, batchsize=32):
    for i in range(0, len(imageset), batchsize):
        yield imageset[i:i + batchsize]

def print_batches(imageset, saveto='', batchsize=32):
    # imageset = np.array(imageset)
    # batches = np.array_split(imageset, len(imageset)//batchsize)
    batches = list(get_batches(imageset, batchsize=8))

    # plot images
    for b, batch in enumerate(batches):
        fig, axs = plt.subplots(4, 8, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)
        axs = axs.ravel()
        for i, img in enumerate(batch):
            axs[i].imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
        plt.show()
        # plt.savefig(saveto + '/batch{}.png'.format(b))
        # plt.close()

# def norm_to_unit_length(data, reg=0.0):
#     # fig, (ax1, ax2, ax3) = plt.subplots(1,3)
#     # ax1.imshow(data, cmap='gray')
#     # ax1.set_title('Input')
#     mean = np.mean(data)
#     data = (data - mean)
#     norm = np.linalg.norm(data)
#     # print('Patch norm: {}'.format(norm))
#     # ax2.imshow(data, cmap='gray')
#     # ax2.set_title('Mean subtraction')
#
#     data = data / (norm + 1e-4)
#     # ax3.imshow(data, cmap='gray')
#     # ax3.set_title('unit norm')
#     # plt.show()
#
#     return data#, [mean, norm]

# img = cv2.imread('./dataset/SDUMLA-HMT/075/left/middle_1.bmp', 0)
# img = compute_properties_image(img, histEq=True, correctStripes=False, correctEdges=False)
# patches64 = extract_patches_image(img, patchSize=(64, 64), slide=(25, 25)).patches
# patches72 = extract_patches_image(img, patchSize=(72, 72), slide=(20, 20)).patches
#
# img64 = np.zeros_like(img.imageNormalised)
# img72 = np.zeros_like(img64)
#
# for p64, l64 in zip(patches64.patches, patches64.locations):
#     rowT, colL = l64
#     rowB = rowT + 64
#     colR = colL + 64
#
#     img64[rowT:rowB, colL:colR] = norm_to_unit_length(p64)
#
# for p72, l72 in zip(patches72.patches, patches72.locations):
#     rowT, colL = l72
#     rowB = rowT + 72
#     colR = colL + 72
#
#     img72[rowT:rowB, colL:colR] = norm_to_unit_length(p72)
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(img64, cmap='gray')
# ax2.imshow(img72, cmap='gray')
# plt.show()

# datapath = './dataset/SDUMLA-HMT/'
# images = read_images_SD(datapath, numSample=1)
# images = compute_properties_image_set(images, histEq=True, correctStripes=False)
# images = extract_patches_image_set(images, patchSize=(64, 64), slide=(18, 18))
# n_patches = 0
# for finger in images:
#     for img in finger:
#         n_patches += len(img.patches.patches)
# print(n_patches)
# #
# # patches = images[2][0].patches.patches
# # print_batches(patches)
# # print()
# #
# im = 0
# for f, finger in enumerate(images):
#     for i, image in enumerate(finger):
#         # if im == 13:
#         #     aux = np.zeros_like(image.imageNormalised)
#         #     upperEdge = image.upperEdge
#         #     lowerEdge = image.lowerEdge
#         #     mask = image.mask
#         #     h, w = aux.shape
#         #     for i in range(w):
#         #         aux[upperEdge[i]][i] = 1
#         #         aux[lowerEdge[i]][i] = 1
#         #     fig, ax = plt.subplots(1,3)
#         #     ax = ax.ravel()
#         #     ax[0].imshow(image.image, cmap='gray')
#         #     ax[1].imshow(aux, cmap='gray')
#         #     ax[2].imshow(mask, cmap='gray')
#         #     plt.show()
#         # im += 1
#         # npatch += len(image.patches.patches)
#         patches = image.patches.patches
#         locations = image.patches.locations
#         img = np.zeros_like(image.image).astype(np.float32)
#         for patch, location in zip(patches, locations):
#             patch = patch - np.mean(patch)
#             norm = np.linalg.norm(patch)
#             patch = patch / (norm + 1e-4)
#             h, w = patch.shape
#             rowT = location[0]
#             rowB = rowT + h
#             colL = location[1]
#             colR = colL + w
#             img[rowT:rowB, colL:colR] = patch
#         fig, ax = plt.subplots(1,2)
#         ax = ax.ravel()
#         ax[0].imshow(image.imageNormalised, cmap='gray')
#         ax[1].imshow(img, cmap='gray')
#         plt.title('Image: {}'.format(im))
#         plt.show()
#         im += 1
# #         plt.savefig('./figures/patch46/finger{}_image{}.png'.format(f, i))
# #         plt.close()
# # print('# patches: {}'.format(npatch))
