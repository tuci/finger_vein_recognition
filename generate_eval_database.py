import numpy as np
import argparse, os
from read_images import Device 
from process_image_set import compute_properties_image_set
from generate_pairs import generate_genuine_pairs, select_object_images
from lmdb_database import make_lmdb
from fine_register_images import horizontal_alignment_windows, vertical_alignment, correct_rotation
from register_images import register_images

# parser command line arguments
parser = argparse.ArgumentParser(description='Evaluation dataset generation')
parser.add_argument('--imagepath', help='Finger image folder', nargs='?', type=str)
parser.add_argument('--dbpath', help='Database folder', nargs='?', type=str)
parser.add_argument('--nSample', help='Number of evaluation subjects', nargs='?', type=int)
parser.add_argument('--nObject', help='Number of nonmated pairs per finger', nargs='?', type=int)
parser.add_argument('--mode', help='data generation mode - train / eval', default='eval', type=str)
parser.add_argument('--device', help='Dataset name', nargs='?', type=str)
parser.add_argument('--finealign', help='If fine align images', action='store_true')
parser.add_argument('--correctEdges', help='Correct Edges', action='store_true')

args = parser.parse_args()

def generate_pairs_partition(partition, num_object=3, finealign=False):
    genuine_pairs = generate_genuine_pairs(partition, finealign)

    imposter_pairs =[]

    for refID in range(len(partition)):
        set = partition[refID]
        for refimage in set:
            object_images = select_object_images(partition, refID, num_object)
            for objimage in object_images:
                if refimage.p == objimage.p and refimage.f == objimage.f:
                    print('Genuine in imposter set!!')
                    return 0
                objimage = register_images(refimage, objimage)
                refpair, objpair = horizontal_alignment_windows([refimage, objimage])
                if finealign:
                    objpair = correct_rotation([refpair, objpair])
                    objpair = vertical_alignment([refpair, objpair])
                    imagepair = [refpair.imageNormalised.astype(np.uint8), 
                                 objpair.imageRegistered.astype(np.uint8), 0]
                else:
                    imagepair = [refpair.imageNormalised.astype(np.uint8), 
                                 objpair.imageRegistered.astype(np.uint8), 0]
                # add image pair to array
                imposter_pairs.append(imagepair)

    return genuine_pairs, imposter_pairs

def main():
    data = Device(device=args.device)
    images = data.read_device(args.imagepath, nSub=args.nSample, mode=args.mode)

    # compute properties
    if args.device == 'UT':
        correctStripes = True
    else:
        correctStripes = False

    images = compute_properties_image_set(images, histEq=True, correctStripes=correctStripes, 
            correctEdges=args.correctEdges)

    # generate evaluation partition
    mated_pairs, nonmated_pairs = generate_pairs_partition(images, num_object=args.nObject, finealign=args.finealign)

    # mated database
    if not os.path.exists(args.dbpath):
        os.makedirs(args.dbpath)
    make_lmdb(mated_pairs, args.dbpath + '/mated/', mapsize=int(1e8))
    # nonmated database
    make_lmdb(nonmated_pairs, args.dbpath + '/nonmated/', mapsize=int(1e8))


if __name__ == '__main__':
    main()


