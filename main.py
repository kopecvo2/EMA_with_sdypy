##

# All important stuff is njuow in PCA of aluminum parts FRF jupyter notebook

import sys

# Append path to sdypy-EMA project from https://github.com/sdypy/sdypy-EMA.git
sys.path.append('C:/Users/pc/PycharmProjects/sdypy-EMA/sdypy') # Change it to path on your computer

import tools as tl # Import of tools made in this project
tl.plt.rcParams['figure.dpi'] = 300
tl.plt.rcParams['savefig.dpi'] = 300
tl.plt.rcParams['figure.figsize'] = [10, 5]

path_to_data = ['C:/Users/pc/OneDrive - České vysoké učení technické v Praze/DATA_D/_GithubProjectData/'
                'EMA_with_sdypy/UFF_with_FRF_aluminum_casting/'] # Change it to path on your computer

approx_nat_freq = [1310, 1800, 3220, 3940, 5540, 6200, 6240, 7150, 7450, 7800, 8450, 8710, 8850, 9000, 9300,
                   9700] # , 5860

part1_1 = tl.ModelEMA(path_to_data[0], 'Scan_odlitek1_testP1_s1.UFF')
part1_1.get_stable_poles()
part1_1.select_closest_poles(approx_nat_freq)

part1_1.reconstruct_avg()

pass