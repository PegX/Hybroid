# s2v_mlp network
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import tensorflow as tf
import numpy as np


# structure2vec
# DE-MF : discriminative embedding using Mean Field


class Network_s2v_mlp:

    def __init__(self,
                 max_lv,
                 T_iterations,
                 learning_rate,
                 l2_reg_lambda,
                 batch_size,
                 logdir
                 ):
        self.max_lv = max_lv
        self.T_iterations = T_iterations
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.batch_size = batch_size
        self.features_size = 64
        self.embedding_size = 64
        self.logdir = logdir
        self.ges_x = []
        self.generateGraphClassificationNetwork()

    def extract_axis_1(self, data, ind):
        """
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """
        ind=tf.nn.relu(ind-1)
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)

        return res

    def create_flattening_array(self, max_nodes, batch_size):
        shape_array = []
        for p in range(0, batch_size):
            for i in range(0, max_nodes):
                shape_array.append([p, i])
        return shape_array

    def create_gather_array(self, max_nodes, batch_size):
        shape_array = []
        for p in range(0, batch_size):
            x = []
            for i in range(0, max_nodes):
                x.append([0, i + p * max_nodes])
            shape_array.append(x)
        return shape_array

    def meanField2(self, input_x, input_adj, name):
        # for batch processing
        W1_tiled = tf.tile(tf.expand_dims(self.W1, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W1_tiled")
        W2_tiled = tf.tile(tf.expand_dims(self.W2, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W2_tiled")

        CONV_PARAMS_tiled = []
        for lv in range(self.max_lv):
            CONV_PARAMS_tiled.append(tf.tile(tf.expand_dims(self.CONV_PARAMS[lv], 0), [tf.shape(input_x)[0], 1, 1],
                                             name=name + "_CONV_PARAMS_tiled_" + str(lv)))

        w1xv = tf.matmul(input_x, W1_tiled, name=name + "_w1xv")
        l = tf.matmul(input_adj, w1xv, name=name + '_l_iteration' + str(1))
        out = w1xv
        for i in range(self.T_iterations - 1):
            ol = l
            lv = self.max_lv - 1
            while lv >= 0:
                with tf.name_scope('cell_' + str(lv)) as scope:
                    node_linear = tf.matmul(ol, CONV_PARAMS_tiled[lv], name=name + '_conv_params_' + str(lv))
                    if lv > 0:
                        ol = tf.nn.relu(node_linear, name=name + '_relu_' + str(lv))
                    else:
                        ol = node_linear
                lv -= 1

            out = tf.nn.tanh(w1xv + ol, name=name + "_mu_iteration" + str(i + 2))
            l = tf.matmul(input_adj, out, name=name + '_l_iteration' + str(i + 2))

        #fi = tf.expand_dims(tf.reduce_sum(out, axis=1, name=name + "_y_potential_reduce_sum"), axis=1,
        fi = tf.expand_dims(tf.reduce_mean(out, axis=1, name=name + "_y_potential_reduce_sum"), axis=1,

                    name=name + "_y_potential_expand_dims")

        graph_embedding = tf.matmul(fi, W2_tiled, name=name + '_graph_embedding')
        return graph_embedding



    def convert_sparse_matrix_to_sparse_tensor(self,X):


        return tf.convert_to_tensor(X)

    def meanField(self, input_x, input_adj, name):

        W1_tiled = self.W1
        W2_tiled = tf.tile(tf.expand_dims(self.W2, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W2_tiled")

        CONV_PARAMS_tiled = []
        for lv in range(self.max_lv):
            CONV_PARAMS_tiled.append(tf.tile(tf.expand_dims(self.CONV_PARAMS[lv], 0), [tf.shape(input_x)[0], 1, 1],name=name + "_CONV_PARAMS_tiled_" + str(lv)))
        w1xv = tf.matmul(input_x, W1_tiled, name=name + "_w1xv")

        adj  = tf.convert_to_tensor(input_adj)
        l = tf.matmul(adj, w1xv, name=name + '_l_iteration' + str(1))
        for i in range(self.T_iterations - 1):
            #print("i",i)
            ol = l
            lv = self.max_lv - 1
            while lv >= 0:
                with tf.name_scope('cell_' + str(lv)) as scope:

                    node_linear = tf.matmul(ol, self.CONV_PARAMS[lv], name=name + '_conv_params_' + str(lv))
                    if lv > 0:
                        ol = tf.nn.relu(node_linear, name=name + '_relu_' + str(lv))
                    else:
                        ol = node_linear
                lv -= 1

            out = tf.nn.tanh(w1xv + ol, name=name + "_mu_iteration" + str(i + 2))

            l = tf.matmul(adj, out, name=name + '_l_iteration' + str(i + 2))

        fi = tf.matmul(tf.expand_dims(out,1), W2_tiled, name=name + '_graph_embedding')
        graph_embedding = fi

        return graph_embedding



    def multilayer_perceptron(self, x):
        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of neurons
        n_hidden_2 = 256 # 2nd layer number of neurons
        n_input = 64   # graph embedding size
        n_classes = 1  # binary classification
        # Store layers weight & bias

        self.w_1 =  tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="w_1")
        self.w_out = tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="w_out")
        self.b_1 = tf.Variable(tf.random_normal([n_hidden_1]), name="b_1")
        self.b_out =  tf.Variable(tf.random_normal([n_classes]), name="b_out")
        
        self.ges_x.append(x)
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, self.w_1), self.b_1)
        sess = tf.Session()
        out_layer = tf.matmul(layer_1, self.w_out) + self.b_out


        return out_layer

    def generateGraphClassificationNetwork(self):

        #self.x_1 = tf.placeholder(tf.int32, [None, None, self.max_instructions],name="x_1")
        self.x_1 = tf.placeholder(tf.float32, [None,64],name="x_1")
        #self.adj_1 = tf.sparse_placeholder(tf.float32, [None, None], name="adj_1")  #
        self.adj_1 = tf.placeholder(tf.float32, [None, None], name="adj_1")  #

        self.y = tf.placeholder(tf.float32, [None], name='y_')

        self.gre = tf.placeholder(tf.float32, name='graphembedding')
        # Euclidean norms; p = 2
        self.norms = []

        l2_loss = tf.constant(0.0)

        self.eps = 0.006
        self.clip_min = 0
        self.clip_max = 1

        # -------------------------------
        #   1. MEAN FIELD COMPONENT
        # -------------------------------

        # 1. parameters for MeanField
        with tf.name_scope('parameters_MeanField'):

            # W1 is a [d,p] matrix, and p is the embedding size as explained above
            self.W1 = tf.Variable(tf.truncated_normal([self.features_size, self.embedding_size], stddev=0.1), name="W1")
            self.norms.append(tf.norm(self.W1))

            # CONV_PARAMSi (i=1,...,n) is a [p,p] matrix. We refer to n as the embedding depth (self.max_lv)
            self.CONV_PARAMS = []
            for lv in range(self.max_lv):
                v = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                name="CONV_PARAMS_" + str(lv))
                self.CONV_PARAMS.append(v)
                self.norms.append(tf.norm(v))

            # W2 is another [p,p] matrix to transform the embedding vector
            self.W2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                  name="W2")
            self.norms.append(tf.norm(self.W2))

        # Mean Field
        with tf.name_scope('MeanField1'):


            self.graph_embedding_1 = tf.nn.l2_normalize(
               tf.squeeze(self.meanField(self.x_1, self.adj_1, "MeanField1")))
            print("graph_embedding : ",self.graph_embedding_1)

        with tf.name_scope("Classifier"):
            self.logits = self.multilayer_perceptron(self.gre)[0]

        # Regularization
        with tf.name_scope("Regularization"):
            l2_loss += tf.nn.l2_loss(self.W1)
            for lv in range(self.max_lv):
                l2_loss += tf.nn.l2_loss(self.CONV_PARAMS[lv])
            l2_loss += tf.nn.l2_loss(self.W2)

        # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.logits, self.y), name="loss")

            self.regularized_loss = self.loss + self.l2_reg_lambda * l2_loss  # regularization

            self.grad, = tf.gradients(self.loss, self.gre)
            self.signed_grad = tf.sign(self.grad)
            self.scaled_signed_grad = self.eps * self.signed_grad
            #self.adv_x = tf.stop_gradient(self.gre + self.scaled_signed_grad)
            self.adv_x = self.gre 
           # self.adv_x = tf.clip_by_value(self.adv_x_1, self.clip_min, self.clip_max)

        # Train step
        with tf.name_scope("Train_Step"):
           self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.regularized_loss)
