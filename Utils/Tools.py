# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 下午9:02
# @Author  : yang chen
import os
import random
import time

import numpy as np
import pandas as pd
import torch
def SetRandomSeed(seed=None):
    if seed == None:
        seed = random.randint(1, 4294967295)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print('seed : {}'.format(seed))
    return seed

def getTime():
    return time.time_ns()

