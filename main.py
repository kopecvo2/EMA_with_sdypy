# Importing newest sdypy project clonned from github
import sys
sys.path.append('C:/Users/pc/PycharmProjects/sdypy-EMA/sdypy/')


import pyuff
import EMA
import numpy as np
import matplotlib.pyplot as plt

uff_file = pyuff.UFF('C:/Users/pc/Desktop/Scan_odlitek1_testP1_s1.UFF')

uff_file.get_set_types()

data = uff_file.read_sets()

acc = EMA.Model(lower=10,
                upper=10000,
                pol_order_high=20,
                frf_from_uff=True)

acc.read_uff('C:/Users/pc/Desktop/Scan_odlitek1_testP1_s1.UFF')



