"""
@File: utils
@Time: 3/18/25 12:30 AM
@Author: liincuan
@Description: 
"""

import glob as glob
import os
import ntpath
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt


def parse_files_from_path(data_path, patterns=('reference', 'pre_gain', 'post_gain')):
    """
    This function searches wav files inside all subdirectories of data_path. Inside every subdir, it parses the files
    names that are uniquely recognized by patterns and assigns their full paths into a returned list.
    :param data_path: string. Relative path to parent directory (e.g. 'test' in attached example).
    :param patterns: tuple of strings. Contains unique file names to be recognized in each subdirectory. In attached
    example the user can use patterns=('reference', 'pre_gain', 'post_gain') or ('ref', 'pre', 'post').
    :return: paths_list: list of lists. Each sublist contains the full paths to the files detected by patterns inside a
    given subdirectory.
    """
    data_list = glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    paths_list = []

    batch_counter = 0
    while batch_counter < len(data_list):
        head = ntpath.split(data_list[batch_counter])[0]
        all_elements = [elem for i, elem in enumerate(data_list) if head in elem]
        elements_to_find = []
        for pattern in patterns:
            matches = [elem for elem in all_elements if pattern in elem]
            if len(matches) != 1:
                return 'Error: Each pattern must match exactly one file in each subdirectory.'
            elements_to_find.append(matches[0])
        paths_list.append(elements_to_find)
        batch_counter += len(all_elements)

    return paths_list


def read_wav_files(path_list, patterns=('reference', 'pre_gain', 'post_gain'), plot=False):
    """
    This function reads all wav files that their paths are contained in path_list, and returns their data as a 2-dim
    numerical array along with their sample frequency.
    :param path_list: list of strings. Contains full paths to wav files. See parse_files_from_path doc for details.
    :param patterns: tuple of strings. See parse_files_from_path doc for details.
    :param plot: boolean. If True, plots content of wav files in path_list by order of patterns.
    :return: wav_array: array of floats. Contains the data samples of the wav files in path_list by order of patterns.
    Number of rows equals the number of samples inside the wav files, number of columns equals the length of patterns.
    :return: fs: int. sample frequency of data samples of the wav files in path_list.
    """

    wav_array = None
    fs_ref = None
    len_wav_ref = None

    for i,path in enumerate(path_list):
        fs, wav_data = wavfile.read(path)
        if fs_ref is not None and fs != fs_ref:
            raise ValueError('Error: All sample frequencies must be identical')
        if len_wav_ref is not None and len(wav_data) != len_wav_ref:
            raise ValueError('Error: All signal lengths must be identical')
        fs_ref = fs
        len_wav_ref = len(wav_data)
        if wav_array is None:
            wav_array = np.empty((wav_data.size, len(patterns)))
        wav_array[:, i] = np.divide(wav_data, pow(2, 15))

    if plot:
        fig, axs = plt.subplots(len(patterns), 1, sharey='col')
        for ax_index in range(len(patterns)):
            axs[ax_index].plot(np.divide(np.arange(len(wav_data)), fs), wav_array[:, ax_index])
            axs[ax_index].set_xlabel('seconds')
            axs[ax_index].set_ylabel(patterns[ax_index])
            axs[ax_index].grid(True)
        fig.tight_layout()
        plt.show()

    return wav_array, fs


def save_and_plot(resl, dsml, path, log_file, save_to_text=True, plot=True):
    """
    This function saves and plots the resl and dsml measures that are derived from the batch of examples inside path.
    :param resl: array of floats. The resl measures derived from the data inside path.
    :param dsml: array of floats. The dsml measures derived from the data inside path.
    :param path: string. The path of subdirectory that contains the batch of examples from which the resl and dsml are
    derived.
    :param log_file: txt file. Contains log of the resl and dsml measures derived from the data inside path.
    :param save_to_text: boolean. If True, write log of the resl and dsml measures to log_file.
    :param plot: boolean. If True, plot the resl and dsml measures.
    :return: log_file: txt file. Contains log of the resl and dsml measures derived from the data inside path.
    """
    evaluation_metrics_to_write = [f'{ntpath.split(path[0])[0]}:\n',
                                   f'RESL: {np.nanmean(resl):.2f} \u00B1 {np.nanstd(resl):.2f} dB \n',
                                   f'DSML: {np.nanmean(dsml):.2f} \u00B1 {np.nanstd(dsml):.2f} dB \n\n']
    if save_to_text:
        log_file.writelines(evaluation_metrics_to_write)
    if plot:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(resl)
        axs[0].set_ylabel('resl')
        axs[0].set_title(ntpath.split(path[0])[0])
        axs[1].plot(dsml)
        axs[1].set_ylabel('dsml')
        axs[1].set_xlabel('frames')
        fig.tight_layout()
        plt.show()
    return log_file
