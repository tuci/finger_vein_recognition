import numpy as np
from ICP_transformation import icp

def compute_register_transformation(upperEdgeRef, lowerEdgeRef, upperEdgeObj,
                                    lowerEdgeObj):
    # COMPUTE_REGISTER_TRANSAFORMATION computes transformation to register object
    # image with a reference image with ICP algorithm
    #
    # Returns:
    #   transformation - transformation that registers the object image
    #       > 2D affine transformation

    nPoints = 2 * len(upperEdgeRef)

    # ICP works in 3D, create 3D point arrays
    # reference image points
    xReference = np.arange(1,nPoints/2+1, dtype=int)
    xReference = np.tile(xReference,2)
    yReference = np.concatenate((upperEdgeRef, lowerEdgeRef), axis=None)
    zReference = np.zeros(nPoints, dtype=int)
    coordinatesReference = np.transpose(
        np.vstack((xReference,yReference,zReference)))

    # object image points
    xObject = np.arange(1,nPoints/2+1, dtype=int)
    xObject = np.tile(xObject,2)
    yObject = np.concatenate((upperEdgeObj,lowerEdgeObj), axis=None)
    zObject = np.zeros(nPoints, dtype=int)
    coordinatesObject = np.transpose(
        np.vstack((xObject,yObject,zObject)))

    ICPTransformation = icp(coordinatesReference,
                                               coordinatesObject)
    transformation = np.zeros((3,3))
    transformation[0:3,0:3] = ICPTransformation[0:3,0:3]
    transformation[0][2] = ICPTransformation[0][3]
    transformation[1][2] = ICPTransformation[1][3]

    return transformation