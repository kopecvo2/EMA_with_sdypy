import sys

import numpy as np

# Append path to sdypy-EMA project from https://github.com/sdypy/sdypy-EMA.git

sys.path.append('C:/Users/pc/PycharmProjects/sdypy-EMA/sdypy')

import tools as tl

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

part = tl.ModelEMA(path_to_data[0], 'Scan_odlitek1_testP1_s1.UFF')

part.get_stable_poles()

part.poles_from_intervals(intervals)

part.reconstruct_avg()

part2 = tl.ModelEMA(path_to_data[0], 'Scan_odlitek4_testP1_s1.UFF')

part2.get_stable_poles()

part2.poles_from_intervals(intervals)

part2.reconstruct_avg()

tl.prettyMAC(part, part2)

# part_old = part
# part_old.name = 'old_part'
#
# part_old.select_closest_poles(approx_nat_freq)
#
# part.reconstruct_avg()
#
# tl.prettyMAC(part, part_old, 'MAC11.png')


part2 = tl.ModelEMA(path_to_data[0] + 'Scan_odlitek1_testP1_s2.UFF')

part2.get_stable_poles()

part2.poles_from_intervals(intervals)

part2.avg_plot()

# part1 = tl.model(path_to_data[0] + 'Scan_odlitek1_testP1_s1.UFF', approx_nat_freq)
#
# tl.reconstruct_avg(part1, approx_nat_freq)
#
# part1.select_poles()

part = tl.model(path_to_data[0] + 'Scan_odlitek5_testP1_s2.UFF', approx_nat_freq, pol_order=200)

tl.poles_from_intervals(part, intervals, plot=False, binsize=10)

tl.reconstruct_avg(part, approx_nat_freq)


part.select_poles()

tl.reconstruct_avg(part2, approx_nat_freq)

part3 = tl.model(path_to_data[0] + 'Scan_odlitek3_testP1_r1.UFF', approx_nat_freq)
part4 = tl.model(path_to_data[0] + 'Scan_odlitek4_testP1_s1.UFF', approx_nat_freq)
part5 = tl.model(path_to_data[0] + 'Scan_odlitek5_testP1_s1.UFF', approx_nat_freq)

tl.reconstruct_avg(part1, approx_nat_freq)
tl.reconstruct_avg(part2, approx_nat_freq)
tl.reconstruct_avg(part3, approx_nat_freq)
tl.reconstruct_avg(part4, approx_nat_freq)
tl.reconstruct_avg(part5, approx_nat_freq)

# reconstruct_scroll(acc)

MAC11 = tl.EMA.tools.MAC(part1.A, part1.A)
MAC12 = tl.EMA.tools.MAC(part1.A, part2.A)
MAC13 = tl.EMA.tools.MAC(part1.A, part3.A)
MAC14 = tl.EMA.tools.MAC(part1.A, part4.A)
MAC15 = tl.EMA.tools.MAC(part1.A, part5.A)

pass
