
import thirdparty.sdypy_EMA.sdypy.EMA as EMA
import thirdparty.sdypy_EMA.sdypy.EMA.stabilization as stabilization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import pyuff  # pyuff is used by EMA
import ctypes
import pandas as pd
import seaborn as sns


def prettyMAC(model1, model2, name_str=None, show_plot=True):
    """

    :param model1: object of class ModelEMA
    :param model2: object of class ModelEMA
    :param name_str: string containing name of figure
    :param show_plot: bool, True if to show the graph
    :return: MAC, pandas.DataFrame of modal assurance criteria MAC of model1 and model2
    """
    MAC = EMA.tools.MAC(model1.model.A, model2.model.A)
    MAC = pd.DataFrame(MAC, columns=np.around(model1.nat_freq).astype(int),
                       index=np.around(model2.nat_freq).astype(int))
    MAC = MAC.round(3)

    fig = plt.figure(facecolor='w', edgecolor='k')
    sns.heatmap(MAC, annot=True, cmap='viridis', cbar=False)
    plt.title('MAC')
    plt.xlabel(model1.name)
    plt.ylabel(model2.name)
    plt.ioff()
    if name_str:
        plt.title('MAC:' + name_str)
        plt.savefig('DataFrame.png')
    if show_plot:
        plt.ion()
        plt.show()
    return MAC


class ModelEMA:

    def __init__(self, path_to_files, name, pol_order=150, binsize=10):
        """

        :param path_to_files: string of path to folder with uff files with measured data
        :param name: string of name of uff file with measured data
        :param pol_order: the highest order of model for stabilization diagram
        :param binsize: size of bin for histogram
        """
        self.pole_list = None
        """list of stable poles in respective bins of frequency histogram"""
        self.bin_vector = None
        """vector of starts of bins of frequency histogram"""
        self.nat_freq = None
        """frequency of poles of identified model"""
        self.pole_inds = None
        """array of indices of poles of identified model"""
        self.peak_indices = None
        """indices of used peaks in frequency histogram"""
        self.freq_histogram = None
        """histogram of number of stable poles in bins of given frequency"""
        self.nat_xi = None
        """damping of poles of identified model"""
        self.path = path_to_files + name
        """path to uff file with data"""
        self.name = name
        """name of uff file with data"""
        self.pol_order = pol_order
        """the highest order of model for stabilization diagram"""
        self.model = self.find_poles()
        """model, object of class sdypy.EMA.Model"""
        self.binsize = binsize
        """size of bin for histogram"""
        self.f_stable = []
        """list of frequencies of stable poles"""
        self.xi_stable = []
        """list of damping of stable poles"""
        self.approx_nat_freq = None
        """approximate frequencies for closest poles method"""

    def find_poles(self):
        """

        Creates EMA stabilization diagram of measured data
        :return: model, object of class sdypy.EMA.Model
        """
        model = EMA.Model(lower=10,
                          upper=10000,
                          pol_order_high=self.pol_order)    # frf_from_uff=True

        model.read_uff(self.path)
        model.get_poles(method='lscf', show_progress=True)

        return model

    def get_stable_poles(self, fn_temp=0.0001, xi_temp=0.05):
        """

        Uses sdypy.EMA.stabilization to find stable poles in stabilization diagram.
        Finds self.f_stable, self.xi_stable - stable pole frequency and damping.
        :param fn_temp: float, coefficient for evaluating pole stability in frequency
        :param xi_temp: float, coefficient for evaluating pole stability in damping
        """
        Nmax = self.pol_order
        # Copied from EMA
        poles = self.model.all_poles
        fn_temp, xi_temp, test_fn, test_xi = stabilization._stabilization(poles, Nmax, err_fn=fn_temp, err_xi=xi_temp)
        # select the stable poles
        b = np.argwhere((test_fn > 0) & ((test_xi > 0) & (xi_temp > 0)))

        mask = np.zeros_like(fn_temp)
        mask[b[:, 0], b[:, 1]] = 1  # mask the unstable poles
        f_stable = fn_temp * mask
        xi_stable = xi_temp * mask
        f_stable[f_stable != f_stable] = 0
        xi_stable[xi_stable != xi_stable] = 0
        self.f_stable = f_stable
        self.xi_stable = xi_stable

        self.model.f_stable = f_stable  # Get rid of it in future

    def poles_from_intervals(self, intervals, plot=False, binsize=30):
        """

        Method finds peaks of frequency histogram of stable poles within given interval and locates most possible pole
        :param intervals: list of lists: [[start freq. interval, end freq. interval, number of expected poles in
        interval],[...],...]
        :param plot: bool, True for plotting results
        :param binsize: size of bins of frequency histogram
        :return:
        """

        self.histo_freq(binsize=binsize)

        histo = 1 * self.freq_histogram
        bin_v = self.bin_vector
        peak_indices = []
        pole_inds = []
        nat_freq = []
        nat_xi = []

        for interval in intervals:
            h_ind = np.argwhere((bin_v >= interval[0]) & (bin_v < interval[1]))
            for i in np.arange(0, interval[2]):
                peak_ind = int(np.argmax(histo[h_ind]) + h_ind[0])
                histo[int(peak_ind - 1):int(peak_ind + 2)] = np.array([0, 0, 0])
                peak_indices.append(peak_ind)
                pole_ind = self.choose_pole_from_bin(self.pole_list[peak_ind][0])
                pole_inds.append(pole_ind)
                nat_freq.append(self.model.pole_freq[pole_ind[0]][pole_ind[1]])
                nat_xi.append(self.model.pole_xi[pole_ind[0]][pole_ind[1]])

        sort = np.argsort(peak_indices)
        self.peak_indices = np.array(peak_indices)[sort]
        self.pole_inds = np.array(pole_inds)[sort]
        self.nat_freq = np.array(nat_freq)[sort]
        self.nat_xi = np.array(nat_xi)[sort]

        if plot:
            fig, ax = plt.subplots()
            ax.stairs(histo[::2],
                      np.append(bin_v[::2], self.model.upper),
                      color='red',
                      fill=True,
                      alpha=0.5)
            ax.stairs(histo[1::2], np.append(bin_v[1::2], self.model.upper),
                      color='red',
                      fill=True,
                      alpha=0.5)
            ax.stairs(self.freq_histogram[::2],
                      np.append(bin_v[::2], self.model.upper),
                      color='dimgray',
                      fill=True,
                      alpha=0.5)
            ax.stairs(self.freq_histogram[1::2],
                      np.append(bin_v[1::2], self.model.upper),
                      color='dimgray',
                      fill=True,
                      alpha=0.5)
            ax.set_ylabel('Histogram of stable poles (red=unused by intervals)')
            plt.show()

        # Rewriting model parameters
        self.model.nat_freq = self.nat_freq
        self.model.nat_xi = self.nat_xi
        self.model.pole_ind = self.pole_inds

    def choose_pole_from_bin(self, ind_poles_in_bin, do_print=False):
        """

        Method chooses most possible pole of given stable poles of close frequency by choosing the one with the most
        possible damping.
        :param ind_poles_in_bin: indices of poles, from which to choose
        :param do_print: bool, True to print messages
        :return: pole_ind - indices of chosen poles
        """
        poles_in_bin = self.f_stable[ind_poles_in_bin[:, 0], ind_poles_in_bin[:, 1]]
        xi_in_bin = self.xi_stable[ind_poles_in_bin[:, 0], ind_poles_in_bin[:, 1]]
        hist, bin_vec = np.histogram(xi_in_bin, bins=10)
        ind_bin = np.argmax(hist)
        arg_sel_xi = np.argwhere((xi_in_bin >= bin_vec[ind_bin]) & (xi_in_bin < bin_vec[ind_bin + 1]))
        mean_xi = np.mean(xi_in_bin[arg_sel_xi])
        arg_pole = np.argmin(np.abs(xi_in_bin - mean_xi))
        model_order = ind_poles_in_bin[arg_pole, 1]
        pole_position = np.argmin(
            ((self.model.pole_xi[model_order] - xi_in_bin[arg_pole]) / xi_in_bin[arg_pole]) ** 2 +
            ((self.model.pole_freq[model_order] - np.mean(poles_in_bin)) / np.mean(poles_in_bin)) ** 2)
        pole_ind = [model_order, pole_position]

        if do_print:
            print(f"expected freq:{np.mean(poles_in_bin)}")
            print(f"found freq:{self.model.pole_freq[model_order][pole_position]}")
            print(f"expected xi:{mean_xi}")
            print(f"found xi:{self.model.pole_xi[model_order][pole_position]}")

        return pole_ind

    def histo_freq(self, binsize=30):
        """

        Method creates histogram of stable poles, the x-axis being frequency of poles
        :param binsize: Size of histogram bin
        :return:
        """
        bin_vector = np.arange(self.model.lower, self.model.upper, binsize / 2)
        histogram = np.array([])
        pole_list = []

        for low_freq in bin_vector:
            ind_poles_in_bin = np.argwhere((self.f_stable >= low_freq) & (self.f_stable < (low_freq + binsize)))
            pole_list.append([ind_poles_in_bin])
            histogram = np.append(histogram, np.size(ind_poles_in_bin, 0))

        self.freq_histogram = histogram
        self.bin_vector = bin_vector
        self.pole_list = pole_list

    def select_closest_poles(self, approx_nat_freq, fn_temp=0.00002, xi_temp=0.05):
        """

        Method uses sdypy.EMA.Model.select_closest_poles to choose from stable poles.
        :param approx_nat_freq: List of natural frequencies, to which the method will find the closest stable poles
        :param fn_temp: float, coefficient for evaluating pole stability in frequency
        :param xi_temp: float, coefficient for evaluating pole stability in damping
        :return:
        """

        self.model.select_closest_poles(approx_nat_freq, fn_temp=fn_temp, xi_temp=xi_temp)
        self.approx_nat_freq = approx_nat_freq

        simplecheck = self.model.nat_freq - approx_nat_freq

        if max(simplecheck) > 100:
            print('Warning: difference between expected and found nat. freq. differences in Hz:')
            print(simplecheck)

        # Getting model parameters
        self.nat_freq = self.model.nat_freq
        self.nat_xi = self.model.nat_xi
        self.pole_inds = self.model.pole_ind

    def reconstruct_avg(self, binsize=None):
        """
        Plots average of magnitude of all FRF, modelled and measured. Uses sdypy.EMA.Model.get_constants method to find
        eigenvectors of the model.
        :param binsize: size of bin of histogram
        :return:
        """
        if not binsize:
            if self.binsize:
                binsize = self.binsize
            else:
                binsize = 30

        if self.freq_histogram is None:
            self.histo_freq(binsize=binsize)

        if not self.freq_histogram.any():
            self.histo_freq(binsize=binsize)

        self.model.get_constants(method='lsfd', f_lower=None)

        reconstructed = np.mean(np.abs(self.model.H), axis=0)
        frequencies = self.model.freq
        measured = np.mean(np.abs(self.model.frf), axis=0)

        fig, ax = plt.subplots()  # figsize=(60000, 20000)

        ax2 = ax.twinx()

        ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
        ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
        ax.legend()

        ax2.stairs(self.freq_histogram[::2],
                   np.append(self.bin_vector[::2], self.model.upper),
                   color='dimgray',
                   fill=True,
                   alpha=0.5,
                   label='histogram')
        ax2.stairs(self.freq_histogram[1::2],
                   np.append(self.bin_vector[1::2], self.model.upper),
                   color='dimgray',
                   fill=True,
                   alpha=0.5,
                   label='histogram')

        ax.set_yscale('log')
        plt.title('FRF average ' + self.name)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('log magnitude ' + self.model.frf_type)
        ax2.set_ylabel('bins with stable poles')

        if self.approx_nat_freq:
            ax.plot(self.approx_nat_freq,
                    np.mean(measured) * np.ones_like(self.approx_nat_freq),
                    'b+',
                    label='approx. nat. freq.')
            ax.plot(self.model.nat_freq,
                    np.mean(measured) * np.ones_like(self.approx_nat_freq),
                    'r+',
                    label='found nat. freq.')
            # plt.legend()
            ax.legend()

        plt.show()

    def reconstruct_scroll(self):
        """
        Function creates interactive canvas with comparation of measured and reconstructed FRFs from model for each FRF.
        :return:
        """

        self.model.get_constants(method='lsfd', f_lower=None)

        def plot(dummy, p):
            """
            plots desired FRF on canvas.
            :param dummy: Unused input
            :param p: Position of scrollbar from 0 to 1-ScrollerHeight
            :return: None
            """

            p_num = int(float(p) // ScrollerHeight)
            scrollbar.set(float(p), float(p) + ScrollerHeight)

            reconstructed = np.abs(self.model.H[p_num, :])
            frequencies = self.model.freq
            measured = np.abs(self.model.frf[p_num, :])

            ax.clear()
            ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
            ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
            ax.set_yscale('log')
            plt.title('FRF ' + str(p_num + 1) + ' ' + self.name)
            plt.xlabel('Frequency [Hz]')
            ax.set_ylabel('log magnitude ' + self.model.frf_type)

            plt.legend()
            canvas.draw()

        # Work with model
        num_of_locations = len(self.model.H[:, 0])
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
