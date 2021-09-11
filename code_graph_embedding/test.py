import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from s2v_classifier_train import S2VTrainer
from parameters import Flags

def run_test():
    flags = Flags()
    print(str(flags))

    trainer = S2VTrainer(flags)
    #trainer.train()
    trainer.testing()

if __name__ == '__main__':
    run_test()
