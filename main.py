##
import sys
import pandas as pd

import numpy as np

# Append path to sdypy-EMA project from https://github.com/sdypy/sdypy-EMA.git

sys.path.append('C:/Users/pc/PycharmProjects/sdypy-EMA/sdypy')

import tools as tl
import stat_tools as stat

import matplotlib.pyplot as plt

#tl.plt.rcParams['figure.figsize'] = [0, 20] # No influence on results
# tl.plt.rcParams['figure.dpi'] = 300

tl.plt.rcParams['figure.figsize'] = [10, 5]

path_to_data = ['C:/Users/pc/OneDrive - České vysoké učení technické v Praze/DATA_D/_GithubProjectData/'
                'EMA_with_sdypy/UFF_with_FRF_aluminum_casting/']


# approx_nat_freq = [1310, 1820, 3220, 3940, 5540, 6200, 6240, 7150, 7450, 7800, 8450, 8710, 8850, 9000, 9300,
#                    9700]    # , 5860

intervals = [[1200, 1450, 1],
             [1450, 2900, 1],
             [2900, 3400, 1],
             [3700, 4300, 1],
             [5200, 5700, 1],
             #[5800, 6000, 1],
             [6000, 6500, 2],
             [7000, 8000, 3],
             [8000, 10000, 6]
             ]

data = pd.DataFrame()

EMAS = []

measurements = []


def proc(path):
    part = tl.ModelEMA(path_to_data[0], path, pol_order=150)

    part.get_stable_poles()

    part.poles_from_intervals(intervals)

    part.reconstruct_avg()

    measurements.append(part)

    return part

p11 = proc('Scan_odlitek1_testP1_s1.UFF')
p12 = proc('Scan_odlitek1_testP1_s2.UFF')
p21 = proc('Scan_odlitek2_testP1_s1.UFF')
p22 = proc('Scan_odlitek2_testP1_s2.UFF')

##

df = pd.DataFrame(columns=['name', 'nat_freq', 'damping', 'eig_vec', 'MAC_diag'])

for meas in measurements:
    eig_vec = meas.model.A.flatten()

    MAC_diag = tl.prettyMAC(p11, meas, show_plot=True).to_numpy().diagonal()

    df.loc[len(df.index)] = [meas.name,
                             meas.nat_freq,
                             meas.nat_xi,
                             eig_vec,
                             MAC_diag]

Labels = [0, 0, 1, 1]
Colors = ['red', 'blue']
target_names = ['meas_1', 'meas_2']
features = ['nat_freq', 'damping']

X = stat.PCA(df, features, Labels, Colors, target_names)

pass
