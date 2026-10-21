import numpy as np
import lmdb, pickle, cv2
from torch.utils.data import Dataset
from process_image_set import compute_edges, extract_roi, adjust_edges

class Image:
    def __init__(self, image, type='ref', fx=1.0, fy=1.0, correctEdges=False):
        if type == 'ref':
            self.imageNormalised = cv2.resize(image, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            self.maskNormalised = compute_mask(self.imageNormalised, correctEdges=correctEdges)
        elif type == 'probe':
            self.imageRegistered = cv2.resize(image, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            self.maskRegistered = compute_mask(self.imageRegistered, correctEdges=correctEdges)

def compute_mask(image, correctEdges):
    [upperEdge, lowerEdge] = compute_edges(image)
    if correctEdges:
        upperEdge, lowerEdge = adjust_edges(upperEdge, lowerEdge)
    # create mask based on edges covering the finger in the
    # original image
    imgHeight, imgWidth = image.shape
    mask = np.transpose(np.zeros((imgHeight, imgWidth), dtype=bool))
    for c in range(imgWidth):
        mask[c][upperEdge[c]:lowerEdge[c]] = True
    mask = np.transpose(mask)

    return mask

class dataloader_miura_search(Dataset):
    def __init__(self, lmdb_root, patchSize=(64, 64), slide=64, scale=(2.5, 1.58), maxshift=(20, 20), 
                 correctEdges=False, mode='train'):
        # open database for reading
        self.env = lmdb.open(lmdb_root, readonly=True)
        self.cursor = list(enumerate(self.env.begin(write=False).cursor()))
        self.nSamples = self.env.stat()['entries']
        self.patchSize = patchSize
        self.slide = slide
        self.fx, self.fy = scale
        self.ch, self.cw = maxshift
        self.correctEdge = correctEdges
        self.mode = mode

    def close_db(self):
        self.env.close()

    def __getitem__(self, index):
        # read image from self.cursor and convert byte data to array
        databin = pickle.loads(self.cursor[index][1][1])

        if self.mode == 'train':
            patch = databin[0]

            patch, _, _ = norm_to_unit_length(np.pad(patch, (1, 0), mode='edge'))

            return np.expand_dims(patch, axis=0).astype(np.float32)

        elif self.mode == 'eval':
            # prepare images
            refgray = Image(databin[0], fx=self.fx, fy=self.fy, correctEdges=self.correctEdge)
            probegray = Image(databin[1], type='probe', fx=self.fx, fy=self.fy, correctEdges=self.correctEdge)

            # extract patches
            refpatches = self.extract_reference_patches(refgray)
            probepatches = self.extract_probe_patches(probegray)

            return refpatches, probepatches

    def print_candidate_pair(self, refpatches):
        images = np.zeros((3, 370, 672))
        for reflist, refloclist in zip(refpatches[0], refpatches[2]):
            for ref, refloc in zip(reflist, refloclist):
                h, w = ref.shape
                rowT_R, colL_R = refloc
                rowB_R = rowT_R + h
                colR_R = colL_R + w
                images[0, rowT_R:rowB_R, colL_R:colR_R] = ref

        return images

    def extract_reference_patches(self, imageset):
        # define patch and location array
        patches = []
        locations = []

        # extract reference patches from finger region
        image = imageset.imageNormalised
        mask = imageset.maskNormalised
        upperEdge, lowerEdge = compute_edges(image)
        # create a dummy mask
        dummymask = np.zeros_like(mask)
        dummymask[min(upperEdge):max(lowerEdge), :] = 1

        # loop over all non-overlapping patches
        [mRow, nColumn] = image.shape

        h, w = self.patchSize
        for rowBottom in range(h, mRow, self.slide):
            rowTop = rowBottom - h
            rowpatches = []
            rowlocations = []
            for columnRight in range(w, nColumn, self.slide):
                columnLeft = columnRight - w
                # check mask
                onDummyROI = np.all(dummymask[rowTop:rowBottom, columnLeft:columnRight])
                onROI = np.all(mask[rowTop:rowBottom, columnLeft:columnRight])
                if ~onDummyROI:
                    continue
                else:
                    if ~onROI:
                        patch = np.full(self.patchSize, np.nan)
                    else:
                        # add patch and its location
                        patch = image[rowTop:rowBottom, columnLeft:columnRight]
                    row = rowTop
                    column = columnLeft
                    rowpatches.append(norm_to_unit_length(np.pad(patch, (1, 0), mode='edge'))[0])
                    rowlocations.append([row, column])
            if len(rowpatches) != 0:
                patches.append(rowpatches)
                locations.append(rowlocations)

        return np.asarray(patches), np.asarray(locations)

    def extract_probe_patches(self, imageset):
        # get roi boundaries
        _, tanUp, tanLow = extract_roi(imageset.maskRegistered)

        # extract probe patches from roi_patch region
        image = imageset.imageRegistered
        mask = imageset.maskRegistered
        maskROI = np.zeros_like(mask)
        maskCopy = np.copy(maskROI)
        maskROI[tanUp + self.ch:tanLow - self.ch, self.cw:-self.cw] = 1
        width = abs(tanUp - tanLow)
        if (width - (2 * self.ch)) <= np.min(tanLow - tanUp)//2:
            ch = width //4
        else:
            ch = self.ch
        if (maskROI.shape[1] - (2 * self.cw)) <= maskROI.shape[1]//2: 
            length = image.shape[1]
            cw = length //4 
        else:
            cw = self.cw

        maskCopy[tanUp + ch:tanLow - ch, cw:-cw] = 1
        maskROI = maskCopy.astype(bool)

        # define patch and location array
        patches = []
        locations = []

        # loop over all non-overlapping patches
        [mRow, nColumn] = image.shape

        h, w = self.patchSize
        for rowBottom in range(h, mRow, self.slide):
            rowTop = rowBottom - h
            rowpatches = []
            rowlocations = []
            for columnRight in range(w, nColumn, self.slide):
                columnLeft = columnRight - w
                # check mask
                onROI = np.all(maskROI[rowTop:rowBottom, columnLeft:columnRight])
                if ~onROI:
                    continue
                # add patch and its location
                patch = image[rowTop:rowBottom, columnLeft:columnRight]
                row = rowTop
                column = columnLeft
                rowpatches.append(norm_to_unit_length(np.pad(patch, (1, 0), mode='edge'))[0])
                rowlocations.append([row, column])
            if len(rowpatches) != 0:
                patches.append(rowpatches)
                locations.append(rowlocations)

        return np.asarray(patches), np.asarray(locations)


    def __len__(self):
        return self.nSamples

def norm_to_unit_length(data, reg=0.0):
    mean = np.mean(data)

    data = (data - mean)
    norm = np.linalg.norm(data)

    data_norm = (data / norm)

    return data_norm, mean, norm


    def __init__(self, lmdb_root):
        # open database for reading
        self.env = lmdb.open(lmdb_root, readonly=True)
        self.cursor = list(enumerate(self.env.begin(write=False).cursor()))
        self.nSamples = self.env.stat()['entries']

    def __getitem__(self, index):
        # read image from self.cursor and convert byte data to array
        databin = pickle.loads(self.cursor[index][1][1])
        features = databin[0][0, :, :, :]
        label = databin[1]
        return features, label

    def __len__(self):
        return self.nSamples