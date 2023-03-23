# Importing newest sdypy project clonned from github
import sys

sys.path.append('C:/Users/pc/PycharmProjects/sdypy-EMA/sdypy/')

import pyuff
import EMA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

# EMA
acc = EMA.Model(lower=10,
                upper=10000,
                pol_order_high=100,
                frf_from_uff=True)

acc.read_uff('C:/Users/pc/Desktop/Scan_odlitek5_testP1_s1.UFF')

acc.get_poles(method='lscf', show_progress=True)

# acc.select_poles()

approx_nat_freq = [1310, 1826, 3220, 3940, 5540, 5860, 6200, 6240, 7150, 7450, 7800, 8450, 8710, 8850, 9000, 9300, 9700]
acc.select_closest_poles(approx_nat_freq)

H, A = acc.get_constants(method='lsfd', f_lower=None)

MAC = acc.autoMAC()

acc.print_modal_data()

simplecheck = acc.nat_freq - approx_nat_freq

print('Expected and found nat. freq. differences in Hz:')
print(simplecheck)


def reconstruct(model):
    """
    Function creates interactive canvas with comparation of measured and reconstructed FRFs from model for each FRF.
    :param model: Object of class EMA.EMA.Model
    :return: None
    """
    def plot(dummy, p):
        """
        plots desired FRF on canvas.
        :param dummy: Unused input
        :param p: Position of scrollbar from 0 to 1-ScrollerHeight
        :return: None
        """

        p_num = int(float(p) // ScrollerHeight)
        scrollbar.set(float(p), float(p) + ScrollerHeight)

        reconstructed = np.abs(model.H[p_num, :])
        frequencies = model.freq
        measured = np.abs(model.frf[p_num, :])

        # fig.clear()
        ax.clear()
        ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
        ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
        ax.set_yscale('log')
        plt.title('FRF '+str(p_num+1))
        plt.xlabel('Frequency [Hz]')
        ax.set_ylabel('log magnitude ' + model.frf_type)

        ax2 = ax.twinx()
        ax2.clear()
        ax2.plot(approx_nat_freq, np.ones_like(approx_nat_freq), 'b+')
        ax2.plot(model.nat_freq, np.ones_like(approx_nat_freq), 'r+')

        plt.legend()
        canvas.draw()

    # Work with model
    num_of_locations = len(model.H[:, 0])
    ScrollerHeight = 1 / num_of_locations

    # Initialize tkinter
    root = tk.Tk()
    fig, ax = plt.subplots()

    canvas = FigureCanvasTkAgg(fig, master=root)

    # Tkinter app
    frame = tk.Frame(root)
    label = tk.Label(text='Use the scrollbar on the right to choose FRF')
    label.pack()

    scrollbar = tk.Scrollbar(root, command=plot)
    scrollbar.pack(side='right', fill='y')
    frame.pack()
    scrollbar.set(0, ScrollerHeight)
    plot(0, 0)
    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(anchor='sw', side='bottom')
    canvas.get_tk_widget().pack(expand=True, fill='both')

    root.mainloop()


reconstruct(acc)

pass
