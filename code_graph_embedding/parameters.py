# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import argparse
import time
import os
import logging

def getLogger(logfile):
    logger = logging.getLogger(__name__)
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr

class Flags:

    def __init__(self):
        parser = argparse.ArgumentParser(description=' cryptoarb.')

        parser.add_argument("-f", "--input_dataset", dest="input_dataset", help = "name of the dataset folder", required=True)
        parser.add_argument("-o", "--output", dest="output_file", help="output directory for logging and models", required=False)
        parser.add_argument("-e", "--embedding_matrix", dest="embedding_matrix", help="file with the embedding matrix for the instructions",required=False)
        parser.add_argument("-j", "--json_asm2id", dest="json_asm2id",help="file with the dictionary of instructions ids", required=False)
        parser.add_argument("-n", "--dbName", dest="db_name", help="Name of the database", required=False)
        parser.add_argument("-ld","--load_dir", dest="load_dir", help="Load the model from directory load_dir", required=False)
        parser.add_argument("-nn","--network_type", help="network type: Arith_Mean, Weighted_Mean, RNN, CCS", required=True, dest="network_type")
        parser.add_argument("-r", "--random", help="if present the network use random embedder", default=False, action="store_true", dest="random_embedding", required=False)
        parser.add_argument("-te","--trainable_embedding", help="if present the network consider the embedding as trainable", action="store_true", dest="trainable_embeddings", default=False)
        parser.add_argument("-cv","--cross_val", help="if present the training is done with cross validiation", default=False, action="store_true", dest="cross_val")

        args = parser.parse_args()
        self.network_type = args.network_type

        if self.network_type == "Annotations":
            self.feature_type = 'acfg'
        elif self.network_type in ["Arith_Mean", "Attention_Mean", "RNN"]:
            self.feature_type = 'cfg'
        else:
            print("ERROR NETWORK NOT FOUND")
            exit(0)

        self.batch_size = 1             # minibatch size (-1 = whole dataset)
        self.num_epochs = 5            # number of epochsi
        self.embedding_size = 64        # dimension of latent layers
        self.learning_rate = 0.001      # init learning_rate
        self.max_lv = 2                 # embedd depth
        self.T_iterations= 2            # max rounds of message passing
        self.l2_reg_lambda = 0          # 0.002 #0.002 # regularization coefficient
        self.cross_val = args.cross_val
        self.cross_val_fold = 5
        self.load_dir = str(args.load_dir)
        self.out_dir = args.output_file
        self.input_dataset = args.input_dataset

        self.seed = 2                   # random seed

        self.reset_logdir()


    def reset_logdir(self):
        # create logdir
        timestamp = str(int(time.time()))
        self.logdir = os.path.abspath(os.path.join(self.out_dir, "runs", timestamp))
        #os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.logdir)

        # create logger
        self.log_file = str(self.logdir)+'/console.log'
        self.logger, self.hdlr = getLogger(self.log_file)

        # create symlink for last_run
        sym_path_logdir = str(self.out_dir)+"/last_run"
        try:
            os.unlink(sym_path_logdir)
        except:
            pass
        try:
            os.symlink(self.logdir, sym_path_logdir)
        except:
            print("\nfailed to create symlink!\n")

    def close_log(self):
        self.hdlr.close()
        self.logger.removeHandler(self.hdlr)
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def __str__(self):
        msg = ""
        msg +="\n  Parameters:\n"
        msg +="\tNetwork_Type: {}\n".format(self.network_type)
        msg +="\tFeature Type: {}\n".format(self.feature_type)
        msg +="\tbatch_size: {}\n".format(self.batch_size)
        msg +="\tnum_epochs: {}\n".format(self.num_epochs)
        msg +="\tlearning_rate: {}\n".format(self.learning_rate)
        msg +="\tmax_lv: {}\n".format(self.max_lv)
        msg +="\tT_iterations: {}\n".format(self.T_iterations)
        msg +="\tl2_reg_lambda: {}\n".format(self.l2_reg_lambda)
        msg +="\tseed: {}\n".format(self.seed)
        return msg
