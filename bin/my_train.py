from __future__ import absolute_import #??

import os
import sys
sys.path.append(os.getcwd())
sys.path.append('..')
from got10k.datasets import * 

from siamfc import SiamFCTracker 

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #多卡情况下默认多卡训练,如果想单卡训练,设置为"0"

if __name__ == '__main__':

    root_dir = os.path.abspath('F:\GOT-10k')#获取当前工作目录

    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    # group=2  是分组卷积, group=1 是原始的alexnet
    tracker = SiamFCTracker(model_path=None) 

    tracker.train_over(seqs) 
    