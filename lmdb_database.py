import numpy as np
import lmdb, os
import pickle


def make_lmdb(images, dbpath, mapsize=None):
    # generate databases for cae-cnn system
    # images - either patches or patch/image pair
    # dbpath - base path to database
    # network - type of network to be the db generated
    #           cae or cnn
    #           if cae is selected only patches are stored
    #           if cnn is selected patches/images are paired and
    #               stored along with their label(genuine - 1, imposter - 0)

    # open database
    if not os.path.exists(dbpath):
        db = lmdb.open(dbpath, map_size=mapsize)
        numkeys = len(images)
        keys = np.random.permutation(numkeys).astype(int)
    else:
        db = lmdb.open(dbpath)
        lendb = db.stat()['entries']
        numkeys = len(images)
        keys = np.random.permutation(numkeys).astype(int) + lendb
    imgindex = 0
    try:
        with db.begin(write=True) as txn:
            for idx, (image, key) in enumerate(zip(images, keys)):
                key = str(key)
                txn.put(key.encode('ascii'), pickle.dumps(image))
                imgindex = idx

    except lmdb.MapFullError:
        curr_limit = db.info()['map_size']
        new_limit = curr_limit * 2
        print('Map size limit reached: {}\tNew limit: {}'.format(curr_limit, new_limit))
        db.set_mapsize(new_limit)

        continue_make_lmdb(images, keys, db, imgindex)

    db.close()

def continue_make_lmdb(data, keys, database, index):
    imgindex = 0
    try:
        with database.begin(write=True) as txn:
            for idx, (image, key) in enumerate(zip(data, keys), start=index+1):
                key = str(key)
                txn.put(key.encode('ascii'), pickle.dumps(image))
                imgindex = idx

    except lmdb.MapFullError:
        curr_limit = database.info()['map_size']
        new_limit = curr_limit * 2
        print('Map size limit reached: {}\tNew limit: {}'.format(curr_limit, new_limit))
        database.set_mapsize(new_limit)

        continue_make_lmdb(data, keys, database, imgindex)

