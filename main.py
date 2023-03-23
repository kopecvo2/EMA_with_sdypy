# Importing newest sdypy project clonned from github
import sys

sys.path.append('C:/Users/pc/PycharmProjects/sdypy-EMA/sdypy/')

import pyuff
import EMA
import numpy as np
import matplotlib.pyplot as plt

# uff_file = pyuff.UFF('C:/Users/pc/Desktop/Scan_odlitek1_testP1_s1.UFF')
#
# uff_file.get_set_types()

# data = uff_file.read_sets()

acc = EMA.Model(lower=10,
                upper=10000,
                pol_order_high=50,
                frf_from_uff=True)

acc.read_uff('C:/Users/pc/Desktop/Scan_odlitek1_testP1_s1.UFF')

acc.get_poles(method='lscf', show_progress=True)

# acc.select_poles()

approx_nat_freq = [1310, 1826, 3220, 3940, 5540, 6200, 6240, 7150, 7450, 7800, 8450, 8750, 8850, 9000, 9300, 9700]
acc.select_closest_poles(approx_nat_freq)

H, A = acc.get_constants(method='lsfd', f_lower=1000)

# FRF = acc.FRF_reconstruct(0)

MAC = acc.autoMAC()

acc.print_modal_data()


def reconstruct(model, location):
    reconstructed = np.abs(model.H[location, :])
    frequencies = model.freq
    measured = np.abs(model.frf[location, :])

    fig, ax = plt.subplots(1, 1)

    ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
    ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
    ax.set_yscale('log')
    plt.legend()
    plt.show()


reconstruct(acc, 0)
reconstruct(acc, 1)
reconstruct(acc, 2)
reconstruct(acc, 3)
reconstruct(acc, 4)
reconstruct(acc, 5)
reconstruct(acc, 6)

pass
