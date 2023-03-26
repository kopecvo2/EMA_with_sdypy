import EMA
import EMA.stabilization as stabilization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import pyuff  # pyuff is used by EMA
import ctypes
import pandas as pd


def model(path, approx_nat_freq):
    """

    :param approx_nat_freq: Expected natural frequencies
    :param path: Path to .UFF file with FRF data
    :return: Object of class EMA.Model
    """
    acc = EMA.Model(lower=10,
                    upper=10000,
                    pol_order_high=100,
                    frf_from_uff=True)

    acc.read_uff(path)

    acc.get_poles(method='lscf', show_progress=True)

    # acc.select_poles()

    acc.select_closest_poles(approx_nat_freq, fn_temp=0.00002, xi_temp=0.05)

    H, A = acc.get_constants(method='lsfd', f_lower=None)

    MAC = acc.autoMAC()

    simplecheck = acc.nat_freq - approx_nat_freq

    if max(simplecheck) > 100:
        print('Warning: difference between expected and found nat. freq. differences in Hz:')
        print(simplecheck)

    return acc


def reconstruct_avg(model, approx_nat_freq, binsize=30):
    """
    Plots average of magnitude of all FRF, modelled and measured.
    :param model: Object of class EMA.Model
    :param approx_nat_freq: Expected natural frequencies
    :return:
    """
    reconstructed = np.mean(np.abs(model.H), axis=0)
    frequencies = model.freq
    measured = np.mean(np.abs(model.frf), axis=0)

    fig, ax = plt.subplots()  # figsize=(60000, 20000)

    ax.clear()
    ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
    ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
    ax.set_yscale('log')
    plt.title('FRF average')
    plt.xlabel('Frequency [Hz]')
    ax.set_ylabel('log magnitude ' + model.frf_type)

    histogram, bin_vector, pole_list = new_histo_freq(model, binsize=binsize)

    ax2 = ax.twinx()
    ax2.clear()
    ax2.plot(approx_nat_freq, 100 * np.ones_like(approx_nat_freq), 'b+', label='approx. nat. freq.')
    ax2.plot(model.nat_freq, 100 * np.ones_like(approx_nat_freq), 'r+', label='found nat. freq.')

    ax2.stairs(histogram[::2], np.append(bin_vector[::2], model.upper), color='dimgray', fill=True, alpha=0.5)
    ax2.stairs(histogram[1::2], np.append(bin_vector[1::2], model.upper), color='dimgray', fill=True, alpha=0.5)

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

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = tk.Tk()
    root.state('zoomed')

    fig, ax = plt.subplots()

    canvas = FigureCanvasTkAgg(fig, master=root)

    # Tkinter app
    frame = tk.Frame(root)
    frame.pack()
    label = tk.Label(text='Use the scrollbar on the right to choose FRF')
    label.pack()

    scrollbar = tk.Scrollbar(root, command=plot)
    scrollbar.pack(side='right', fill='y')
    scrollbar.set(0, ScrollerHeight)
    plot(0, 0)
    toolbar = NavigationToolbar2Tk(canvas, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(anchor='sw', side='bottom')
    canvas.get_tk_widget().pack(expand=True, fill='both')

    root.mainloop()


def prettyMAC(model1, model2):
    MAC = EMA.tools.MAC(model1.A, model2.A)
    MAC = pd.DataFrame(MAC, columns=np.around(model1.nat_freq).astype(int),
                       index=np.around(model2.nat_freq).astype(int))
    MAC = MAC.round(3)
    MAC = MAC.style.background_gradient(axis=None)
    MAC = MAC.format(precision=3)

    return MAC


def histo_freq(model, binsize=10):
    f_window = 50
    Nmax = model.pol_order_high
    fn_temp = 0.0001
    xi_temp = 0.05

    bins = np.arange(model.lower, model.upper + binsize, binsize)

    poles = model.all_poles
    fn_temp, xi_temp, test_fn, test_xi = stabilization._stabilization(
        poles, Nmax, err_fn=fn_temp, err_xi=xi_temp)
    # select the stable poles
    b = np.argwhere((test_fn > 0) & ((test_xi > 0) & (xi_temp > 0)))

    mask = np.zeros_like(fn_temp)
    mask[b[:, 0], b[:, 1]] = 1  # mask the unstable poles
    f_stable = fn_temp * mask
    xi_stable = xi_temp * mask
    f_stable[f_stable != f_stable] = 0
    xi_stable[xi_stable != xi_stable] = 0

    pass


def new_histo_freq(model, binsize=100):
    """

    :param model: Object of class EMA.EMA.Model
    :param binsize: Size of histogram bin
    :return: histogram, binvector, pole_list
    histogram: number of stable poles in respective bin
    bin_vector: vector of starts of bins
    pole_list: list of arrays with indices to model.f_stable of stable poles in respective bin
    """
    bin_vector = np.arange(model.lower, model.upper, binsize / 2)
    histogram = np.array([])
    pole_list = []

    poles = model.f_stable

    for low_freq in bin_vector:
        ind_poles_in_bin = np.argwhere((poles >= low_freq) & (poles < (low_freq + binsize)))
        pole_list.append([ind_poles_in_bin])
        histogram = np.append(histogram, np.size(ind_poles_in_bin, 0))

    return histogram, bin_vector, pole_list


def poles_from_intervals(model, intervals, plot=False):
    old_histo, bin_v, pole_list = new_histo_freq(model, binsize=30)

    histo = 1 * old_histo
    peak_indices = []

    for interval in intervals:
        h_ind = np.argwhere((bin_v >= interval[0]) & (bin_v < interval[1]))
        for i in np.arange(0, interval[2]):
            peak_ind = int(np.argmax(histo[h_ind]) + h_ind[0])
            histo[int(peak_ind - 1):int(peak_ind + 2)] = np.array([0, 0, 0])
            peak_indices.append(peak_ind)
            choose_pole_from_bin(model, pole_list[peak_ind][0])

    if plot:
        fig, ax = plt.subplots()
        ax.stairs(histo[::2], np.append(bin_v[::2], model.upper), color='red', fill=True, alpha=0.5)
        ax.stairs(histo[1::2], np.append(bin_v[1::2], model.upper), color='red', fill=True, alpha=0.5)
        ax.stairs(old_histo[::2], np.append(bin_v[::2], model.upper), color='dimgray', fill=True, alpha=0.5)
        ax.stairs(old_histo[1::2], np.append(bin_v[1::2], model.upper), color='dimgray', fill=True, alpha=0.5)
        ax.set_ylabel('Histogram of stable poles (red=unused by intervals)')
        plt.show()

    pass


def choose_pole_from_bin(model, ind_poles_in_bin):
    poles_in_bin = model.f_stable[ind_poles_in_bin[:, 0], ind_poles_in_bin[:, 1]]
    xi_in_bin = model.xi_stable[ind_poles_in_bin[:, 0], ind_poles_in_bin[:, 1]]
    hist, bin_vec = np.histogram(xi_in_bin, bins=10)
    ind_bin = np.argmax(hist)
    arg_sel_xi = np.argwhere((xi_in_bin >= bin_vec[ind_bin]) & (xi_in_bin < bin_vec[ind_bin+1]))
    mean_xi = np.mean(xi_in_bin[arg_sel_xi])
    arg_pole = np.argmin(np.abs(xi_in_bin-mean_xi))
    model_order = ind_poles_in_bin[arg_pole, 1]
    pole_position = np.argmin(np.abs(model.pole_xi[model_order]-mean_xi))
    pole_ind = [model_order, pole_position]
    print(f"expected freq:{np.mean(poles_in_bin)}")
    print(f"found freq:{model.pole_freq[model_order][pole_position]}")

    return pole_ind

