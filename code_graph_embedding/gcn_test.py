from sklearn import svm

from networkx.classes.digraph import DiGraph
import networkx as nx


import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
from random import shuffle
from progressbar import *

sys.path.append('%s/../apk_graph_corpus/' % os.path.dirname(os.path.realpath(__file__)))



import tensorflow as tf



#import gcn as GCN

class Analysis:
    """ A class to run a classification experiment """

    def __init__(self, dirs, labels, split, max_files=0, max_node_size=0,
                 precomputed_matrix="",  y="", fnames=""):

        self.split = split
        self.X = []
        #self.Y = np.array([])
        self.Y = []
        self.fnames = []
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.clf = ""
        self.out = []
        self.dirs = dirs
        # self.roc = 0
        self.rocs = []
        self.auc = 0
        self.b = 0
        self.feature_vector_times = []
        self.label_dist = np.zeros(2**15)
        self.sample_sizes = []
        self.neighborhood_sizes = []
        self.class_dist = np.zeros(15)
        self.predictions = []
        self.true_labels = []
        self.features_size = 64
        self.embedding_size = 64
        self.max_lv = 2
        self.norms = []
        self.T_iterations = 3


        #f_vector = np.load("embedding_matrx_x64.npy")

        if precomputed_matrix:
            # Load the y labels and file names from zip pickle objects.
            print("Loading matrix...")
            self.X = np.load(precomputed_matrix)
            print("[*] matrix loaded"        )
            self.Y = np.load(y)
            print("[*] labels loaded")
            self.fnames = np.load(fnames)
            print("[*] file names loaded")

        else:
            # loop over dirs
            paths = ["dirs/malware", "dirs/goodware"]
            for d in zip(paths, labels):
                print(zip(paths, labels))
                files = self.read_files(d[0], max_files)
                print("Loading samples in dir {0} with label {1}".format(d[0],
                                                                         d[1]))
                widgets = ['Unpickling... : ',
                           Percentage(), ' ',
                           Bar(marker='#', left='[', right=']'),
                           ' ', ETA(), ' ']
                pbar = ProgressBar(widgets=widgets, maxval=len(files))
                pbar.start()
                progress = 0
                count = 0

                #print "Starting analysis..."
                # load labels and feature vectors
                for f in files:
                    #print(f)
                    try:
                        g = DiGraph()
                        g = np.load(f)
                        size = g.number_of_nodes()
                        if size > 200000:
                            continue
                        else:
                            print("size")
                            print(size)
                            print("The number of node %d" % size)
                            adj = nx.adjacency_matrix(g)
                            print("adj matrix:",adj.shape)
                            x_obs = np.zeros([size,128])
                            node_idx = []
                            node_name = []
                            node_label = []
                            idx = -1
                            for node in iter(g.nodes()):
                                idx += 1
                                node_idx.append(idx)
                                node_name.append(node)
                                node_label.append(g.node[node]["label"])

                            node_name_idx = dict(zip(node_name,node_idx))
                            node_idx_label = dict(zip(node_idx,node_label))
                            self.X.append(g)
                          #self.Y = np.append(self.Y,[int(d[1])])
                            self.Y.append([int(d[1])])
                            self.fnames.append(f)
                            count+= 1
                            print("Handing %d files",count)

                    except  Exception, e:
                        print e
                        print("err: {0}".format(f))

    def save_data(self):
        """ Store pz objects for the data matrix, the labels and
            the name of the original samples so that they can be used
            in a new experiment without the need to extract all
            features again
        """
        print("[*] Saving labels, data matrix and file names...")
        np.save("X.npy", self.X)
        np.save("Y.npy", self.Y)
        np.save("fnames.npy", self.fnames)
        print(len(self.X),len(self.Y),len(self.fnames))


    def read_files(self, d, max_files=0):
        """ Return a random list of N files with a certain extension in dir d

        Args:
            d: directory to read files from
            file_extension: consider files only with this extension
            max_files: max number of files to return. If 0, return all files.

        Returns:
            A list of max_files random files with extension file_extension
            from directory d.
        """

        files = []
        for f in os.listdir(d):
            files.append(os.path.join(d, f))
        shuffle(files)

        # if max_files is 0, return all the files in dir
        if max_files == 0:
            max_files = len(files)
            print(max_files)

        files = files[:max_files]

        #`print "the size of the dataset is %d" % len(files)
        return files

class DataLoad:
    def __init__(self,x,y,fnames):
        self.x = x
        self.y = y
        self.fnames = fnames
        self.split = 0.8
        print(self.x,self.y)

    def data_load(self):
        # Load the y labels and file names from zip pickle objects.
        print("Loading graph...")
        self.X = np.load(self.x)
        print("[*] graph loaded"        )
        self.Y = np.load(self.y)
        print("[*] labels loaded")
        self.fnames = np.load(self.fnames)
        print("[*] file names loaded")

        X_train,Y_train,X_test,Y_test = self.randomize_dataset()
        return X_train,Y_train,X_test,Y_test


    def randomize_dataset(self):
        """ Randomly split the dataset in training and testing sets
        """

        n = len(self.Y)
        m = len(self.X)
        print "Len of output %d" % n
        print "Len of output %d" % m

        train_size = int(self.split * n)
        print "Len of training %d" % train_size
        index = range(n)
        shuffle(index)
        print(index)
        train_index = sorted(index[:train_size])
        test_index = sorted(index[train_size:])

        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        #self.X_train = self.X[train_index, :]
        #self.X_test = self.X[test_index, :]
        for i in train_index:
            self.X_train.append(self.X[i])
            self.Y_train.append(self.Y[i])

        for i in test_index:
            self.X_test.append(self.X[i])
            self.Y_test.append(self.Y[i])

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.Y_train = np.array(self.Y_train)
        self.Y_test =  np.array(self.Y_test)

        return self.X_train,self.Y_train,self.X_test,self.Y_test
