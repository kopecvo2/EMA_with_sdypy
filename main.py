import sys

# Append path to sdypy-EMA project from https://github.com/sdypy/sdypy-EMA.git

sys.path.append('C:/Users/pc/PycharmProjects/sdypy-EMA/sdypy/')

import pyuff
import EMA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

path_to_data = ['C:/Users/pc/OneDrive - České vysoké učení technické v Praze/DATA_D/_GithubProjectData/EMA_with_sdypy/'
                'UFF_with_FRF_aluminum_casting/']


approx_nat_freq = [1310, 1820, 3220, 3940, 5540, 5860, 6200, 6240, 7150, 7450, 7800, 8450, 8710, 8850, 9000, 9300,
                   9700]


def model(path, approx_nat_freq):
    """

    :param approx_nat_freq: Expected natural frequencies
    :param path: Path to .UFF file with FRF data
    :return: Object of class EMA.Model
    """
    acc = EMA.Model(lower=10,
                    upper=10000,
                    pol_order_high=200,
                    frf_from_uff=True)

    acc.read_uff(path)

    acc.get_poles(method='lscf', show_progress=True)

    # acc.select_poles()

    acc.select_closest_poles(approx_nat_freq)

    H, A = acc.get_constants(method='lsfd', f_lower=None)

    MAC = acc.autoMAC()

    acc.print_modal_data()

    simplecheck = acc.nat_freq - approx_nat_freq

    print('Expected and found nat. freq. differences in Hz:')
    print(simplecheck)

    return acc


def reconstruct_avg(model, approx_nat_freq):
    """
    Plots average of magnitude of all FRF, modelled and measured.
    :param model: Object of class EMA.Model
    :param approx_nat_freq: Expected natural frequencies
    :return:
    """
    reconstructed = np.mean(np.abs(model.H), axis=0)
    frequencies = model.freq
    measured = np.mean(np.abs(model.frf), axis=0)

    fig, ax = plt.subplots()

    ax.clear()
    ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
    ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
    ax.set_yscale('log')
    plt.title('FRF average')
    plt.xlabel('Frequency [Hz]')
    ax.set_ylabel('log magnitude ' + model.frf_type)

    ax2 = ax.twinx()
    ax2.clear()
    ax2.plot(approx_nat_freq, np.ones_like(approx_nat_freq), 'b+')
    ax2.plot(model.nat_freq, np.ones_like(approx_nat_freq), 'r+')

    plt.legend()
    plt.show()


def reconstruct_scroll(model):
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

        ax.clear()
        ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
        ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
        ax.set_yscale('log')
        plt.title('FRF ' + str(p_num + 1))
        plt.xlabel('Frequency [Hz]')
        ax.set_ylabel('log magnitude ' + model.frf_type)

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


part1 = model(path_to_data[0] + 'Scan_odlitek1_testP1_s1.UFF', approx_nat_freq)
part2 = model(path_to_data[0] + 'Scan_odlitek2_testP1_s1.UFF', approx_nat_freq)
part3 = model(path_to_data[0] + 'Scan_odlitek3_testP1_r1.UFF', approx_nat_freq)
part4 = model(path_to_data[0] + 'Scan_odlitek4_testP1_s1.UFF', approx_nat_freq)
part5 = model(path_to_data[0] + 'Scan_odlitek5_testP1_s1.UFF', approx_nat_freq)

reconstruct_avg(part1, approx_nat_freq)
reconstruct_avg(part2, approx_nat_freq)
reconstruct_avg(part3, approx_nat_freq)
reconstruct_avg(part4, approx_nat_freq)
reconstruct_avg(part5, approx_nat_freq)

# reconstruct_scroll(acc)

MAC11 = EMA.tools.MAC(part1.A, part1.A)
MAC12 = EMA.tools.MAC(part1.A, part2.A)
MAC13 = EMA.tools.MAC(part1.A, part3.A)
MAC14 = EMA.tools.MAC(part1.A, part4.A)
MAC15 = EMA.tools.MAC(part1.A, part5.A)

pass
