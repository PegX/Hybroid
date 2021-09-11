from sklearn import svm
from networkx.classes.digraph import DiGraph
import networkx as nx
import numpy as np
import os
import sys
from random import shuffle
from progressbar import *
sys.path.append('%s/../apk_graph_corpus/' % os.path.dirname(os.path.realpath(__file__)))
import tensorflow as tf



class PrepareDataset(object):
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.X_list = []
        self.Y_list = []
        self.split = 0.8
        #self.trainDataDir = "dirs_training"
        self.count = 0


    def getGraphFeatures(self, file, label):
        try:
            g = DiGraph()
            g = np.load(file)
            size = g.number_of_nodes()
            if size < 80000:
                adj = nx.adjacency_matrix(g)
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

                self.X_list.append(g)
                self.Y_list.append(label)

                self.count+= 1
                print("Handing %d files", self.count)
                return
        except  Exception, e:
            print e
            print("err: {0}".format(file))


    def walkThroughgrahs(self, directory, label):
        for root, dirs, files in os.walk(directory):
            for file in files:
                self.getGraphFeatures(os.path.join(root, file), label)
        return

    def prepare_training(self, path):
        for root, dirs, files in os.walk(path):
            for directory in dirs:
                if "malware" in directory:
                    self.walkThroughgrahs(os.path.join(root, directory), -1)
                if "goodware" in directory:
                    self.walkThroughgrahs(os.path.join(root, directory), 1)

        X_train,Y_train,X_test,Y_test = self.split_dataset()
        return X_train,Y_train,X_test,Y_test

    def split_dataset(self):

        x_len = len(self.X_list)
        y_len = len(self.Y_list)

        train_size = int(self.split * y_len)
        test_size = int((1 - self.split) * y_len)

        print "Len of training %d" % train_size
        print "Len of testing %d" % test_size
        index = range(y_len)
        shuffle(index)
        train_index = sorted(index[:train_size])
        test_index = sorted(index[train_size:])

        for i in train_index:
            self.X_train.append(self.X_list[i])
            self.Y_train.append(self.Y_list[i])

        for i in test_index:
            self.X_test.append(self.X_list[i])
            self.Y_test.append(self.Y_list[i])

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.Y_train = np.array(self.Y_train)
        self.Y_test =  np.array(self.Y_test)

        return self.X_train, self.Y_train, self.X_test, self.Y_test
