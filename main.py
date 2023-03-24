import sys

# Append path to sdypy-EMA project from https://github.com/sdypy/sdypy-EMA.git

sys.path.append('C:/Users/vojta/PycharmProjects/sdypy-EMA/sdypy')

import tools as tl

tl.plt.rcParams['figure.figsize'] = [50, 20]
tl.plt.rcParams['figure.dpi'] = 300

path_to_data = ['C:/Users/vojta/OneDrive - České vysoké učení technické v Praze/DATA_D/_GithubProjectData/'
                'EMA_with_sdypy/UFF_with_FRF_aluminum_casting/']


approx_nat_freq = [1310, 1820, 3220, 3940, 5540, 5860, 6200, 6240, 7150, 7450, 7800, 8450, 8710, 8850, 9000, 9300,
                   9700]



part1 = tl.model(path_to_data[0] + 'Scan_odlitek1_testP1_s1.UFF', approx_nat_freq)

tl.reconstruct_scroll(model=part1)
tl.reconstruct_avg(part1, approx_nat_freq)
MAC11 = tl.EMA.tools.MAC(part1.A, part1.A)

part2 = tl.model(path_to_data[0] + 'Scan_odlitek2_testP1_s1.UFF', approx_nat_freq)
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
