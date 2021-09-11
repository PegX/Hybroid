"""Generic object pickler and compressor

This module saves and reloads compressed representations of generic Python
objects to and from the disk.
"""
import pickle as cPickle
import pickle
#import cPickle as pickle
import gzip


def save(object, filename, bin=1):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, bin))
    file.close()


def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while 1:
            data = file.read()
            if data == "":
                    break
            buffer += str(data)
    object = pickle.loads(buffer)
    file.close()
    #print "loading file type %s" % type(object)
    return object


if __name__ == "__main__":
    import sys
    import os.path

    class Object:
            x = 7
            y = "This is an object."

    filename = sys.argv[1]
    if os.path.isfile(filename):
            o = load(filename)
            print("Loaded %s" % o)
    else:
            o = Object()
            save(o, filename)
            print("Saved %s" % o)

