import EMA
import EMA.stabilization as stabilization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import pyuff  # pyuff is used by EMA
import ctypes
import pandas as pd
import dataframe_image as dfi
import seaborn as sns


# def model(path, approx_nat_freq, pol_order=150):
#     """
#
#     :param approx_nat_freq: Expected natural frequencies
#     :param path: Path to .UFF file with FRF data
#     :return: Object of class EMA.Model
#     """
#     acc = EMA.Model(lower=10,
#                     upper=10000,
#                     pol_order_high=pol_order,
#                     frf_from_uff=True)
#
#     acc.read_uff(path)
#
#     acc.get_poles(method='lscf', show_progress=True)
#
#     # acc.select_poles()
#
#     acc.select_closest_poles(approx_nat_freq, fn_temp=0.00002, xi_temp=0.05)
#
#     simplecheck = acc.nat_freq - approx_nat_freq
#
#     if max(simplecheck) > 100:
#         print('Warning: difference between expected and found nat. freq. differences in Hz:')
#         print(simplecheck)
#
#     return acc


# def reconstruct_avg(model, approx_nat_freq=None, binsize=30):
#     """
#     Plots average of magnitude of all FRF, modelled and measured.
#     :param model: Object of class EMA.Model
#     :param approx_nat_freq: Expected natural frequencies
#     :return:
#     """
#     model.get_constants(method='lsfd', f_lower=None)
#
#     reconstructed = np.mean(np.abs(model.H), axis=0)
#     frequencies = model.freq
#     measured = np.mean(np.abs(model.frf), axis=0)
#
#     fig, ax = plt.subplots()  # figsize=(60000, 20000)
#
#     ax.clear()
#     ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
#     ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
#     ax.set_yscale('log')
#     plt.title('FRF average' + model.name)
#     plt.xlabel('Frequency [Hz]')
#     ax.set_ylabel('log magnitude ' + model.frf_type)
#
#     histogram, bin_vector, pole_list = histo_freq(model.lower, model.upper, model.f_stable, binsize=binsize)
#
#     ax2 = ax.twinx()
#     ax2.clear()
#
#     if approx_nat_freq:
#         ax2.plot(approx_nat_freq, 100 * np.ones_like(approx_nat_freq), 'b+', label='approx. nat. freq.')
#         ax2.plot(model.nat_freq, 100 * np.ones_like(approx_nat_freq), 'r+', label='found nat. freq.')
#
#     ax2.stairs(histogram[::2], np.append(bin_vector[::2], model.upper), color='dimgray', fill=True, alpha=0.5)
#     ax2.stairs(histogram[1::2], np.append(bin_vector[1::2], model.upper), color='dimgray', fill=True, alpha=0.5)
#
#     plt.legend()
#     plt.show()


# def reconstruct_scroll(model):
#     """
#     Function creates interactive canvas with comparation of measured and reconstructed FRFs from model for each FRF.
#     :param model: Object of class EMA.EMA.Model
#     :return: None
#     """
#
#     model.get_constants(method='lsfd', f_lower=None)
#
#     def plot(dummy, p):
#         """
#         plots desired FRF on canvas.
#         :param dummy: Unused input
#         :param p: Position of scrollbar from 0 to 1-ScrollerHeight
#         :return: None
#         """
#
#         p_num = int(float(p) // ScrollerHeight)
#         scrollbar.set(float(p), float(p) + ScrollerHeight)
#
#         reconstructed = np.abs(model.H[p_num, :])
#         frequencies = model.freq
#         measured = np.abs(model.frf[p_num, :])
#
#         ax.clear()
#         ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
#         ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)
#         ax.set_yscale('log')
#         plt.title('FRF ' + str(p_num + 1))
#         plt.xlabel('Frequency [Hz]')
#         ax.set_ylabel('log magnitude ' + model.frf_type)
#
#         plt.legend()
#         canvas.draw()
#
#     # Work with model
#     num_of_locations = len(model.H[:, 0])
#     ScrollerHeight = 1 / num_of_locations
#
#     # Initialize tkinter
#
#     ctypes.windll.shcore.SetProcessDpiAwareness(1)
#     root = tk.Tk()
#     root.state('zoomed')
#
#     fig, ax = plt.subplots()
#
#     canvas = FigureCanvasTkAgg(fig, master=root)
#
#     # Tkinter app
#     frame = tk.Frame(root)
#     frame.pack()
#     label = tk.Label(text='Use the scrollbar on the right to choose FRF')
#     label.pack()
#
#     scrollbar = tk.Scrollbar(root, command=plot)
#     scrollbar.pack(side='right', fill='y')
#     scrollbar.set(0, ScrollerHeight)
#     plot(0, 0)
#     toolbar = NavigationToolbar2Tk(canvas, pack_toolbar=False)
#     toolbar.update()
#     toolbar.pack(anchor='sw', side='bottom')
#     canvas.get_tk_widget().pack(expand=True, fill='both')
#
#     root.mainloop()


def prettyMAC(model1, model2, name_str=None, show_plot=True):
    MAC = EMA.tools.MAC(model1.model.A, model2.model.A)
    MAC = pd.DataFrame(MAC, columns=np.around(model1.nat_freq).astype(int),
                       index=np.around(model2.nat_freq).astype(int))
    MAC = MAC.round(3)
    # MAC = MAC.style.background_gradient(axis=None)
    # MAC = MAC.format(precision=3)

    # figure = dfi.export(MAC, name_str)

    fig = plt.figure(facecolor='w', edgecolor='k')
    sns.heatmap(MAC, annot=True, cmap='viridis', cbar=False)
    plt.title('MAC')
    plt.xlabel(model1.name)
    plt.ylabel(model2.name)
    if name_str:
        plt.title('MAC:' + name_str)
        plt.savefig('DataFrame.png')
    if show_plot:
        plt.show()
    return MAC


# def histo_freq(model, binsize=10):
#     f_window = 50
#     Nmax = model.pol_order_high
#     fn_temp = 0.0001
#     xi_temp = 0.05
#
#     bins = np.arange(model.lower, model.upper + binsize, binsize)
#
#     poles = model.all_poles
#     fn_temp, xi_temp, test_fn, test_xi = stabilization._stabilization(
#         poles, Nmax, err_fn=fn_temp, err_xi=xi_temp)
#     # select the stable poles
#     b = np.argwhere((test_fn > 0) & ((test_xi > 0) & (xi_temp > 0)))
#
#     mask = np.zeros_like(fn_temp)
#     mask[b[:, 0], b[:, 1]] = 1  # mask the unstable poles
#     f_stable = fn_temp * mask
#     xi_stable = xi_temp * mask
#     f_stable[f_stable != f_stable] = 0
#     xi_stable[xi_stable != xi_stable] = 0
#
#     pass


# def histo_freq(lower, upper, poles, binsize=30):
#     """
#
#     :param model: Object of class EMA.EMA.Model
#     :param binsize: Size of histogram bin
#     :return: histogram, binvector, pole_list
#     histogram: number of stable poles in respective bin
#     bin_vector: vector of starts of bins
#     pole_list: list of arrays with indices to model.f_stable of stable poles in respective bin
#     """
#     bin_vector = np.arange(lower, upper, binsize / 2)
#     histogram = np.array([])
#     pole_list = []
#
#     for low_freq in bin_vector:
#         ind_poles_in_bin = np.argwhere((poles >= low_freq) & (poles < (low_freq + binsize)))
#         pole_list.append([ind_poles_in_bin])
#         histogram = np.append(histogram, np.size(ind_poles_in_bin, 0))
#
#     return histogram, bin_vector, pole_list


# def poles_from_intervals(model, intervals, plot=False, binsize=30):
#     old_histo, bin_v, pole_list = new_histo_freq(model, binsize=binsize)
#
#     histo = 1 * old_histo
#     peak_indices = []
#     pole_inds = []
#     nat_freq = []
#     nat_xi = []
#
#     for interval in intervals:
#         h_ind = np.argwhere((bin_v >= interval[0]) & (bin_v < interval[1]))
#         for i in np.arange(0, interval[2]):
#             peak_ind = int(np.argmax(histo[h_ind]) + h_ind[0])
#             histo[int(peak_ind - 1):int(peak_ind + 2)] = np.array([0, 0, 0])
#             peak_indices.append(peak_ind)
#             pole_ind = choose_pole_from_bin(model, pole_list[peak_ind][0])
#             pole_inds.append(pole_ind)
#             nat_freq.append(model.pole_freq[pole_ind[0]][pole_ind[1]])
#             nat_xi.append(model.pole_xi[pole_ind[0]][pole_ind[1]])
#
#     sort = np.argsort(peak_indices)
#     peak_indices = np.array(peak_indices)[sort]
#     pole_inds = np.array(pole_inds)[sort]
#     nat_freq = np.array(nat_freq)[sort]
#     nat_xi = np.array(nat_xi)[sort]
#
#     if plot:
#         fig, ax = plt.subplots()
#         ax.stairs(histo[::2], np.append(bin_v[::2], model.upper), color='red', fill=True, alpha=0.5)
#         ax.stairs(histo[1::2], np.append(bin_v[1::2], model.upper), color='red', fill=True, alpha=0.5)
#         ax.stairs(old_histo[::2], np.append(bin_v[::2], model.upper), color='dimgray', fill=True, alpha=0.5)
#         ax.stairs(old_histo[1::2], np.append(bin_v[1::2], model.upper), color='dimgray', fill=True, alpha=0.5)
#         ax.set_ylabel('Histogram of stable poles (red=unused by intervals)')
#         plt.show()
#
#     # Rewriting model parameters
#     model.nat_freq = nat_freq
#     model.nat_xi = nat_xi
#     model.pole_ind = pole_inds
#
#     pass


# def choose_pole_from_bin(model, ind_poles_in_bin):
#     poles_in_bin = model.f_stable[ind_poles_in_bin[:, 0], ind_poles_in_bin[:, 1]]
#     xi_in_bin = model.xi_stable[ind_poles_in_bin[:, 0], ind_poles_in_bin[:, 1]]
#     hist, bin_vec = np.histogram(xi_in_bin, bins=10)
#     ind_bin = np.argmax(hist)
#     arg_sel_xi = np.argwhere((xi_in_bin >= bin_vec[ind_bin]) & (xi_in_bin < bin_vec[ind_bin + 1]))
#     mean_xi = np.mean(xi_in_bin[arg_sel_xi])
#     arg_pole = np.argmin(np.abs(xi_in_bin - mean_xi))
#     model_order = ind_poles_in_bin[arg_pole, 1]
#     pole_position = np.argmin(((model.pole_xi[model_order] - xi_in_bin[arg_pole]) / xi_in_bin[arg_pole]) ** 2 +
#                               ((model.pole_freq[model_order] - np.mean(poles_in_bin)) / np.mean(poles_in_bin)) ** 2)
#     pole_ind = [model_order, pole_position]
#
#     # print(f"expected freq:{np.mean(poles_in_bin)}")
#     # print(f"found freq:{model.pole_freq[model_order][pole_position]}")
#     # print(f"expected xi:{mean_xi}")
#     # print(f"found xi:{model.pole_xi[model_order][pole_position]}")
#
#     return pole_ind


class ModelEMA:

    def __init__(self, path_to_files, name, pol_order=150, binsize=10):
        self.pole_list = None
        self.bin_vector = None
        self.nat_freq = None
        self.pole_inds = None
        self.peak_indices = None
        self.freq_histogram = None
        self.nat_xi = None
        self.path = path_to_files + name
        self.name = name
        self.pol_order = pol_order
        self.model = self.find_poles()
        self.binsize = binsize
        self.f_stable = []
        self.xi_stable = []
        self.approx_nat_freq = None

    def find_poles(self):
        model = EMA.Model(lower=10,
                          upper=10000,
                          pol_order_high=self.pol_order,
                          frf_from_uff=True)

        model.read_uff(self.path)
        model.get_poles(method='lscf', show_progress=True)

        return model

    def get_stable_poles(self, fn_temp=0.0001, xi_temp=0.05):
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

    # def histo_freq(self, binsize=100):
    #     """
    #
    #     :param model: Object of class EMA.EMA.Model
    #     :param binsize: Size of histogram bin
    #     :return: histogram, binvector, pole_list
    #     histogram: number of stable poles in respective bin
    #     bin_vector: vector of starts of bins
    #     pole_list: list of arrays with indices to model.f_stable of stable poles in respective bin
    #     """
    #     bin_vector = np.arange(self.model.lower, self.model.upper, binsize / 2)
    #     histogram = np.array([])
    #     pole_list = []
    #
    #     poles = self.f_stable
    #
    #     for low_freq in bin_vector:
    #         ind_poles_in_bin = np.argwhere((poles >= low_freq) & (poles < (low_freq + binsize)))
    #         pole_list.append([ind_poles_in_bin])
    #         histogram = np.append(histogram, np.size(ind_poles_in_bin, 0))
    #
    #     return histogram, bin_vector, pole_list

    def poles_from_intervals(self, intervals, plot=False, binsize=30):

        # old_histo, bin_v, pole_list = histo_freq(self.model.lower,
        #                                          self.model.upper,
        #                                          self.f_stable,
        #                                          binsize=binsize)
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

        pass

    def choose_pole_from_bin(self, ind_poles_in_bin, do_print=False):
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

        :param model: Object of class EMA.EMA.Model
        :param binsize: Size of histogram bin
        :return: histogram, binvector, pole_list
        histogram: number of stable poles in respective bin
        bin_vector: vector of starts of bins
        pole_list: list of arrays with indices to model.f_stable of stable poles in respective bin
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

        # return histogram, bin_vector, pole_list

    def select_closest_poles(self, approx_nat_freq, fn_temp=0.00002, xi_temp=0.05):

        self.model.select_closest_poles(approx_nat_freq, fn_temp=fn_temp, xi_temp=xi_temp)
        self.approx_nat_freq = approx_nat_freq

        simplecheck = self.model.nat_freq - approx_nat_freq

        if max(simplecheck) > 100:
            print('Warning: difference between expected and found nat. freq. differences in Hz:')
            print(simplecheck)

        return

    def reconstruct_avg(self, binsize=None):
        """
        Plots average of magnitude of all FRF, modelled and measured.
        :param model: Object of class EMA.Model
        :param approx_nat_freq: Expected natural frequencies
        :return:
        """
        if not binsize:
            if self.binsize:
                binsize = self.binsize
            else:
                binsize = 30

        if not self.freq_histogram.any():
            self.histo_freq(binsize=binsize)

        self.model.get_constants(method='lsfd', f_lower=None)

        reconstructed = np.mean(np.abs(self.model.H), axis=0)
        frequencies = self.model.freq
        measured = np.mean(np.abs(self.model.frf), axis=0)

        fig, ax = plt.subplots()  # figsize=(60000, 20000)

        # ax.clear()
        ax2 = ax.twinx()
        # ax2.clear()

        ax.plot(frequencies, measured, 'dimgray', label='measured', linewidth=1.0)
        ax.plot(frequencies, reconstructed, 'r--', label='reconstructed', linewidth=1.0)

        ax2.stairs(self.freq_histogram[::2],
                   np.append(self.bin_vector[::2], self.model.upper),
                   color='dimgray',
                   fill=True,
                   alpha=0.5)
        ax2.stairs(self.freq_histogram[1::2],
                   np.append(self.bin_vector[1::2], self.model.upper),
                   color='dimgray',
                   fill=True,
                   alpha=0.5)

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
            plt.legend()

        plt.show()

    def reconstruct_scroll(self):
        """
        Function creates interactive canvas with comparation of measured and reconstructed FRFs from model for each FRF.
        :param model: Object of class EMA.EMA.Model
        :return: None
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
