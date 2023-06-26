import pickle
import os
import h5py
import tables
import numpy as np


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def txt_load(in_file):
    with open(in_file, "rb") as f:
        content = f.readlines()
    content = [x.strip().decode("utf-8") for x in content]
    return content


def recursive_glob(searchroot='.', searchstr=''):
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    print("search for {0} in {1}".format(searchstr, searchroot))
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if searchstr in filename]
    f.sort()
    return f


def recursive_glob2(searchroot='.', searchstr1='', searchstr2=''):
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    print("search for {} and {} in {}".format(searchstr1, searchstr2, searchroot))
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if (searchstr1 in filename and searchstr2 in filename)]
    f.sort()
    return f


def loadlistfile(filename):
    '''
    load the h5 file names from a .list
    :param filename: .list file
    :return: list of h5 file names
    '''
    filelist = []
    infile = open(filename, 'r')
    for line in infile:
        line = line.strip()
        if len(line) > 0:
            filelist.append(line)
    infile.close()
    return filelist

def load_h5_subgroup(h5filename, key):
    basename = os.path.basename(h5filename)
    group = None
    if '.h5' in basename:
        f = h5py.File(h5filename.strip(), 'r')
        group = f[key]
        f.close()
    return group


def load_h5_pred(h5predfilename):
    basename = os.path.basename(h5predfilename)
    if '.h5' in basename:
        f = h5py.File(h5predfilename.strip(), 'r')
        img = f['pred'].value
        f.close()
    return img


def load_h5_content(h5filename, only_shape=False):
    '''
    load the data of the h5 as a dictionary with keys and images (ndarrays)
    :param h5filename: h5 file name
    :param only_shape:
    :return: a dictionary with {key: image (ndarray), ...}
    '''
    ret_val = []
    basename = os.path.basename(h5filename)
    if '.h5' in basename:
        f = h5py.File(h5filename.strip(), 'r')
        # List all keys
        keys = list(f.keys())
        if only_shape:
            h5data = f['dataL']#f[keys[0]]
            ret_val = h5data.shape
        else:
            imgs = []
            for key in keys:
                h5data = f[key]
                imgs.append(h5data.value)
            ret_val = dict(zip(keys, imgs))
        f.close()
    return ret_val


def loadimglist(listfilename):
    '''
    list h5 data from a .list file sorting with the keys
    :param listfilename: .list file name
    :return: dictionary list [{key1: image (ndarray), key2: image, ..., key4: }]
    '''
    h5files = loadlistfile(listfilename)
    h5diclist = []
    for h5file in h5files:
        imgs = load_h5_content(h5file)
        h5diclist.append(imgs)
    return h5diclist


def load_h5_content_shapes(h5files, listfilename=""):
    if h5files is None:
        h5files = loadlistfile(listfilename)
    shapes = []
    for h5file in h5files:
        shape = load_h5_content(h5file, True)
        shapes.append(shape)
    return shapes

