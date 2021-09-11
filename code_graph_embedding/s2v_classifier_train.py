import tensorflow as tf
import random
import sys, os
import numpy as np
from sklearn import metrics
import matplotlib
import networkx as nx
from sklearn import svm

from networkx.classes.digraph import DiGraph
import networkx as nx

#import matplotlib.pyplot as plt
from random import shuffle
import pz 
from progressbar import *

sys.path.append('%s/../apk_graph_corpus/' % os.path.dirname(os.path.realpath(__file__)))

import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from classifier import Network_s2v_mlp
from gcn_test import DataLoad
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
from scipy import interp
from prepare_matrix import PrepareDataset
import time

class S2VTrainer:

    def __init__(self,flags):
        self.max_lv = flags.max_lv
        self.num_epochs = flags.num_epochs
        self.learning_rate = flags.learning_rate
        self.l2_reg_lambda = flags.l2_reg_lambda
        self.T_iterations = flags.T_iterations
        self.seed = flags.seed
        self.batch_size = flags.batch_size
        self.session = None
        self.network_type = flags.network_type
        self.cross_val = flags.cross_val
        self.logdir = flags.logdir
       # self.logger = flags.logger
        self.num_checkpoints = 1
        self.path_dataset = flags.input_dataset
        random.seed(self.seed)
        np.random.seed(self.seed)


    def createNetwork(self):

        if self.network_type == "Arith_Mean":

            self.network = Network_s2v_mlp(
                max_lv=self.max_lv,
                T_iterations=self.T_iterations,
                learning_rate=self.learning_rate,
                l2_reg_lambda=self.l2_reg_lambda,
                batch_size=self.batch_size,
                logdir = self.logdir
               # logger= self.logger
            )

    def train(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as g:
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )
            sess = tf.Session(config=session_conf)

            # Sets the graph-level random seed.
            tf.set_random_seed(self.seed)

            self.createNetwork()

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # TensorBoard
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", self.network.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(self.logdir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary])
            test_summary_dir = os.path.join(self.logdir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(self.logdir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

            #dataload = DataLoad("X.npy","Y.npy","fnames.npy")
            #X_train,Y_train,X_test,Y_test = dataload.data_load()
            #dataload = PrepareDataset()
            #X_train,Y_train, X_test,Y_test = dataload.prepare_training("dirss")


            begin = time.time()
            step = 0
            totalAccuracy = 0.0
            totalPrecision = 0.0
            totalFScore = 0.0
            totalRecall = 0.0
            pathdataset = "/home/peng/androzoo/CICAndMal2017-fcg-originalName_Train_Test/train/"
            for epoch in range(0, self.num_epochs):
            #for epoch in range(30):
                epoch_msg = ""
                epoch_msg += "  epoch: {}\n".format(epoch)

                epoch_loss = 0

# -------------------------TRAIN ---------------------------------------------

                n_batch=0
                true_count = 0
                total = 0
                predictions = []
                labels = []
                tprs_train=[]
                aucs_train=[]
                fprs_test=[]
                aucs_test=[]
                mean_fpr = np.linspace(0,1,100)
                skipped_counter = 0
                all_loss = 0
                counter = 1
                print"################### TRAIN FOR EPOCH {} ################# ".format(epoch)
                #for root, dirs, files in os.walk(self.path_dataset):
                for root, dirs, files in os.walk(path_dataset):
                    num =len(files)
                    for directory in dirs:
                        if "malware" in directory:
                            for root, dirs, files in os.walk(self.path_dataset + directory):
                                for file in files:
                                    g = DiGraph()
                                    g = pz.load(os.path.join(root, file))
                                    node_label = []
                                   #print("number of nodes", len(X_train[i].nodes()))

                                    if len(g.nodes())>5:
                                        adj_1 = nx.adjacency_matrix(g)
                                        for node in iter(g.nodes()):
                                            node_label.append(np.array(g.node[node]["label"]))

                                    x_1 = node_label
                                    x_1 = np.array(x_1)
                                    print("counter", counter)

                                    #if counter < num*0.95:
                                    #    y = 1
                                    #else:
                                    #    y = -1
                                    y = 1

                                    y = np.array([y])
                                    print("file name", file,len(g.nodes()))
                                    counter +=1


                                    #if type(x_1) == np.ndarray:
                                    if len(g.nodes())>5:
                                        if adj_1.shape[0] > 80000:
                                            print("Skipped file because of nodes", file)
                                            skipped_counter += 1
                                            print("skipped_apk",skipped_counter)
                                            continue
                                        else:
                                            adj_1 = np.array(adj_1.todense())
                                            total += 1
                                            feed_dict = {
                			                        self.network.x_1: x_1,
                			                        self.network.adj_1:adj_1,
                			                        self.network.y: y}
                                            ge1 = sess.run(self.network.graph_embedding_1, feed_dict=feed_dict)
                                            feed_dict2 = {
                                                         self.network.gre: ge1,
                                                         self.network.y: y
                                                    }

                                            summaries, _, loss, norms, logits = sess.run(
                                                                    [train_summary_op, self.network.train_step, self.network.loss, self.network.norms, self.network.logits],
                                                    feed_dict=feed_dict2)
                                           # all_loss +=

                                            #if np.multiply(y,np.tanh(logits)) > 0:
                                                #print("file with positive result", file)
                                            #    true_count +=1
                                             #   print("training: positive results",true_count,"times","in ", total)
                                            #else:
                                                #print("file with negative result", file)

                                            if np.tanh(logits) < 0:
                                                #print("file with negative result", file)
                                                predict = -1
                                            else:
                                                predict = 1
                                        #print(len(predictions))
                                            print("label",y,"predict",np.tanh(logits))
                                            predictions.append(predict)
                                            labels.append(y)
                                    else:
                                        continue

                        if "goodware" in directory:
                            counter = 0
                            #self.walkThroughgrahs(os.path.join(root, directory), 1)
                            for root, dirs, files in os.walk(self.path_dataset +  directory):
                                num = len(files)
                                for file in files:
                                    g = DiGraph()
                                    #g = np.load(os.path.join(root, file))
                                    g = pz.load(os.path.join(root, file))
                                    node_label = []
                                   #print("number of nodes", len(X_train[i].nodes()))
                                    if len(g.nodes())>5:
                                        adj_1 = nx.adjacency_matrix(g)
                                        for node in iter(g.nodes()):
                                            node_label.append(np.array(g.node[node]["label"]))
                                    #adj_1 = nx.adjacency_matrix(g)
                                    #for node in iter(g.nodes()):
                                    #    node_label.append(np.array(g.node[node]["label"]))
                                    x_1 = node_label
                                    x_1 = np.array(x_1)
                                    #if counter < num*0.95:
                                    #    y = -1
                                    #else:
                                    #    y = 1
                                    y = -1

                                    counter +=1
                                    
                                    y = np.array([y])
                                    print("file name", file,len(g.nodes()))
                                    #if type(x_1) == np.ndarray:
                                    if len(g.nodes())>5:
                                        if adj_1.shape[0] > 80000:
                                            print("Skipped file because of nodes", file)
                                            skipped_counter += 1
                                            print("skipped_apk",skipped_counter)
                                            continue
                                        else:
                                            adj_1 = np.array(adj_1.todense())
                                            total += 1
                                            feed_dict = {
                			                        self.network.x_1: x_1,
                			                        self.network.adj_1:adj_1,
                			                        self.network.y: y}
                                            
                                            ge1 = sess.run(self.network.graph_embedding_1, feed_dict=feed_dict)
                                            feed_dict2 = {
                                                         self.network.gre: ge1,
                                                         self.network.y: y
                                                    }

                                            summaries, _, loss, norms, logits = sess.run(
                				                    [train_summary_op, self.network.train_step, self.network.loss, self.network.norms, self.network.logits],
                                                    feed_dict=feed_dict2)


                                          #  if np.multiply(y,np.tanh(logits)) > 0:
                                                #print("file with positive result", file)
                                           #     true_count +=1
                                               # print("training: positive results",true_count,"times","in ",total)
                                                #print("file with negative result", file)
                                            if np.tanh(logits) < 0:
                                                #print("file with negative result", file)
                                                predict = -1
                                            else:
                                                predict = 1
                                        #print(len(predictions))
                                            print("label",y,"predict",np.tanh(logits))
                                            predictions.append(predict)
                                            labels.append(y)
                                    else:
                                        continue



                    #print("graph",i,"with label",y,"predicted value",logits,"predicted value2",np.tanh(logits),"total number",total)
                end = time.time()
                print("time", end-begin)
                print"################### TRAINING RESULT FOR EPOCH {} ####################".format(epoch)
                print("Skipped training samples: ", skipped_counter)
                #epoch_loss = all_loss / len(files)
                #epoch_msg += "\ttrain_loss: {}\n".format(epoch_loss)
                #print("Epoch number", str(epoch))
                tn, fn, fp, tp = confusion_matrix(labels, predictions).ravel()
                #print("metrics", str(tn) , str(fn), str(fp), str(tp))
                print("Accuracy", accuracy_score(labels, predictions))
                print("f1_score",f1_score(labels, predictions))
                print("recall_score",recall_score(labels, predictions))
                print("precision_score",precision_score(labels, predictions))
                totalAccuracy += accuracy_score(labels, predictions)
                totalRecall += recall_score(labels, predictions)
                totalFScore += f1_score(labels, predictions)
                totalPrecision += precision_score(labels, predictions)
            saver.save(sess, self.logdir+'sall', global_step=1000)
                #print("\ttrain_loss: {}\n".format(epoch_loss))

                #print("total number",total,"positive number",true_count)

                    # ----------------------#
                    #         TEST  	    #
                    # ----------------------#

                    # TEST

            true_count = 0
            total = 0

    def testing(self):
        total = 0
        #counter2_skipped = 0
        self.createNetwork()
         # TensorBoard
         # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.network.loss)
         # Restore the trained model and do the testing steps
        #with tf.Session() as sess:
        sess = tf.Session()
             # Initialize all variables
        sess.run(tf.global_variables_initializer())
             # Test summaries
        test_summary_op = tf.summary.merge([loss_summary])
        test_summary_dir = os.path.join(self.logdir, "summaries","test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        #new_saver = tf.train.import_meta_graph('logout2/runs/ournewmodel.meta')
        #new_saver.restore(sess, 'logout2/runs/ournewmodel')
        #new_saver.restore(sess,tf.train.latest_checkpoint('./'))
        #new_saver.restore(sess, self.logdir+'code_similary_model')
        #new_saver=tf.train.import_meta_graph("./logout2/runs/1575063221ournewmodel-1000.meta")
        #new_saver.restore(sess,"./logout2/runs/1575063221ournewmodel-1000")
        new_saver=tf.train.import_meta_graph("./logout2/runs/1583254921sall-1000.meta")
        new_saver.restore(sess,"./logout2/runs/1583254921sall-1000")

        mean_fpr = np.linspace(0,1,100)
        print("############################## TESTING #################################")
        for a in range(5):
            predictions = []
            labels = []
            predictionsArr = []
            labelsArr = []
            true_count = 0
            all_loss = 0
            skipped_counter =0 
            totalAccuracy =0.0
            totalRecall = 0.0
            totalFScore = 0.0
            totalPrecision =0.0 
            epoch_msg = ""

            epoch_loss = 0

            n_batch=0
            total = 0
            predictions = []
            labels = []
            tprs_train=[]
            aucs_train=[]
            fprs_test=[]
            aucs_test=[]
            mean_fpr = np.linspace(0,1,100)

            for root, dirs, files in os.walk(self.path_dataset + "testing_dirs"):
                for directory in dirs:
                    if "malware" in directory:
                        for root, dirs, files in os.walk(self.path_dataset + "testing_dirs/" + directory):
                            for file in files:
                                g = DiGraph()
                                g = pz.load(os.path.join(root, file))
                                node_label = []
                                if len(g.nodes())>5:
                                    #print("number of nodes", len(X_train[i].nodes()))
                                    adj_1 = nx.adjacency_matrix(g)
                                    for node in iter(g.nodes()):
                                        node_label.append(np.array(g.node[node]["label"]))
 
                                x_1 = node_label
                                x_1 = np.array(x_1)

                                y = 1
                                y = np.array([y])
                                #print("file name", file)
                                print("file name", file,len(g.nodes()))

                                #if type(x_1) == np.ndarray:
                                if len(g.nodes())>5:
                                    if adj_1.shape[0] > 80000:
                                        #print("Skipped file because of nodes", file)
                                        skipped_counter += 1
                                        print("skipped_apk",skipped_counter)
                                        continue
                                    else:
                                        adj_1 = np.array(adj_1.todense())
                                        total += 1
                                        feed_dict = {
                			                    self.network.x_1: x_1,
                			                    self.network.adj_1:adj_1,
                			                    self.network.y: y}

                                        ge1 = sess.run(self.network.graph_embedding_1, feed_dict=feed_dict)
                                        feed_dict2 = {
                                                         self.network.gre: ge1,
                                                         self.network.y: y
                                                    }
                                        summaries, _, loss, norms, ge1, logits = sess.run(
                				    [test_summary_op, self.network.train_step, self.network.loss, self.network.norms, self.network.graph_embedding_1, self.network.logits],
#                				    [test_summary_op, self.network.train_step, self.network.loss, self.network.norms,  self.network.logits],
                                        feed_dict=feed_dict2)
                                            #all_loss += loss
                                            #print("value of y", str(y))
                                            #print("value of logits", np.tanh(logits))
                                            # if np.multiply(y,np.tanh(logits)) > 0:
                                            # true_count +=1
                                            # print("training: positive results",true_count,"times","in total",total)
                                            # print("file with positive result", file)
                                            #else:
                                            #print("file with negative result", file)
                                        if np.tanh(logits) < 0:
                                            predict = -1
                                        else:
                                            predict = 1
                                        print("label",y,"predict",np.tanh(logits))
                                        predictions.append(predict)
                                        labels.append(y)
                                        predictionsArr.append(np.tanh(logits))
                                        labelsArr.append(y)

                                else:
                                    continue

                    if "goodware" in directory:
                        #self.walkThroughgrahs(os.path.join(root, directory), 1)
                        for root, dirs, files in os.walk(self.path_dataset + "testing_dirs/" + directory):
                            for file in files:
                                #print("hello file g")
                                g = DiGraph()
                                g = pz.load(os.path.join(root, file))
                                node_label = []
                                    #print("number of nodes", len(X_train[i].nodes()))
                                if len(g.nodes())>5:
                                    adj_1 = nx.adjacency_matrix(g)
                                    #adj_1 = np.array(adj_1.todense())
                                    adj_1 = np.array(adj_1.todense()) 
                                    for node in iter(g.nodes()):
                                        node_label.append(np.array(g.node[node]["label"]))
                                    x_1 = node_label
                                    x_1 = np.array(x_1)
                                    y = -1
                                    y = np.array([y])
                                #print("file name", file)
                                    print("file name", file,len(g.nodes()))
                                #if type(x_1) == np.ndarray:
                 #               if len(g.nodes())>5:
                                    if adj_1.shape[0] > 80000:

                                        skipped_counter += 1
                                        print("skipped_apk",skipped_counter)
                                        continue
                                    else:
                                        #adj_1 = np.array(adj_1.todense())
                                        print("adj_1",adj_1.shape,adj_1)
                                        total += 1
                                        feed_dict = {
                			                    self.network.x_1: x_1,
                			                    self.network.adj_1:adj_1,
                			                    self.network.y: y}
                                        ge1 = sess.run(self.network.graph_embedding_1, feed_dict=feed_dict)
                                        feed_dict2 = {
                                                         self.network.gre: ge1,
                                                         self.network.y: y
                                                    }

                                    #all_loss = 0
                                        summaries, _, loss, norms, ge1, logits = sess.run(
                				    [test_summary_op, self.network.train_step, self.network.loss, self.network.norms, self.network.graph_embedding_1, self.network.logits],
                                    feed_dict=feed_dict2)

                                        if np.tanh(logits) < 0:
                                            predict = -1
                                        else:
                                            predict = 1
                                    #print("predictions length", len(predictions))
                                        predictions.append(predict)
                                        labels.append(y)
                                        predictionsArr.append(np.tanh(logits))
                                        labelsArr.append(y)
                                        print("label",y,"predict",np.tanh(logits))
                                else:
                                    continue



            print"###############  TESTING SCORE FOR ROUND {} ################".format(a)
            #print("Skipped testing samples", counter2_skipped)
            tn, fn, fp, tp = confusion_matrix(labels, predictions).ravel()
            print("tn, fn, fp, tp",tn, fn, fp, tp)
            print("Accuracy", accuracy_score(labels, predictions))
            print("f1_score",f1_score(labels, predictions))
            print("recall_score",recall_score(labels, predictions))
            print("precision_score",precision_score(labels, predictions))
            totalAccuracy += accuracy_score(labels, predictions)
            totalRecall += recall_score(labels, predictions)
            totalFScore += f1_score(labels, predictions)
            totalPrecision += precision_score(labels, predictions)


            #print("graph",i,"with label",y,"predicted value",logits,"predicted value2",np.tanh(logits),"total number",total)
            #print("predicted y1",logits,"predicted y2",np.tanh(logits))
            #epoch_loss = all_loss / len(files)
            #epoch_msg += "\ttrain_loss: {}\n".format(epoch_loss)
            #print("\ttrain_loss: {}\n".format(epoch_loss))

            fpr,tpr,thresholds = roc_curve(labels,predictions)
            tprs_train.append(interp(mean_fpr, fpr, tpr))
            tprs_train[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs_train.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3)
            #print("total number",total,"positive number",true_count)


        print("################### AVERAGE TESTING SCORE #####################")
        print ('Total Accuracy: ' + str(totalAccuracy / 5))
        print ('Total Recall: ' + str(totalRecall / 5))
        print ('Total Fscore: ' + str(totalFScore / 5))
        print ('Total Precision: ' + str(totalPrecision / 5))
        mean_tpr = np.mean(tprs_train, axis=0)
        np.save("mean_ptr.pz",mean_tpr)
        np.save("predictionsArr",predictionsArr)
        np.save("labelsArr",labelsArr)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs_train)
        #plt.plot(mean_fpr, mean_tpr, 'r',label=r'Mean ROC with (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
        plt.plot(mean_fpr, mean_tpr, 'r', alpha=.8)
        print("mean_auc",mean_auc,"std_auc",std_auc)

        std_tpr = np.std(tprs_train, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of Graph-based detector')
        #plt.legend(loc="lower right")
        #plt.show()
        plt.savefig("results.png")


    def Test(self):
        self.createNetwork()
         # TensorBoard
         # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.network.loss)
         # Restore the trained model and do the testing steps
        with tf.Session() as sess:
             # Initialize all variables
            sess.run(tf.global_variables_initializer())
             # Test summaries
            test_summary_op = tf.summary.merge([loss_summary])
            test_summary_dir = os.path.join(self.logdir, "summaries","test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            new_saver = tf.train.import_meta_graph('logout2/runs/ournewmodel.meta')
#new_saver.restore(sess,tf.train.latest_checkpoint('./'))
             #new_saver.restore(sess, self.logdir+'code_similary_model')
            new_saver.restore(sess, 'logout2/runs/ournewmodel')
#new_saver.restore(sess,tf.train.latest_checkpoint('./'))
