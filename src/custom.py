from utils import *

import torch


def custom_wavelet_choices(): # for 10 wavelet levels 
    o1 = [0,1,2,3,4,5,6,7,8,9]
    o2 = [0,2,4,6,8]
    o3 = [0,1,2,3,4]
    o4 = [5,6,7,8,9]
    return [o1,o2,o3,o4]








