"""
@File: evaluation_metrics
@Time: 3/18/25 2:24 PM
@Author: liincuan
@Description: 
"""

import numpy as np
from numpy import add, subtract, multiply, divide


def calc_resl_dsml(wav_array, fs, window_size_ms=20, hop_size_ms=10, lambda_reg=1e-8, compensate=True):
    """
    :param wav_array: array of floats. Contains the data samples of the wav files to be analyzed by the measures. See
    utils.read_wav_files doc for details.
    :param fs: int. Sample frequency of data samples of the wav files in wav_array.
    :param window_size_ms: float. Size of analysis window in milliseconds.
    :param hop_size_ms: float. Hop size between adjacent analysis windows in milliseconds.
    :param lambda_reg: float. Regularization term to avoid divergence of measures.
    :param compensate: boolean. If True, apply gain compensation to measures.
    :return: resl: array of floats. Contains array of the resl values of every analysis windows.
    :return: dsml: array of floats. Contains array of the dsml values of every analysis windows.
    """
    resl, dsml = [], []
    (window_size_samples, hop_size_samples) = (window_size_ms * fs // 1000, hop_size_ms * fs // 1000)

    speech_reference = wav_array[:, 0]
    pre_gain_speech = wav_array[:, 1]
    post_gain_speech = wav_array[:, 2]
    segment_first_sample = 0
    segment_last_sample = segment_first_sample + window_size_samples - 1
    while segment_last_sample < len(pre_gain_speech):
        speech_reference_segment, pre_gain_speech_segment, post_gain_speech_segment = \
            speech_reference[slice(segment_first_sample, segment_last_sample + 1)], \
            pre_gain_speech[slice(segment_first_sample, segment_last_sample + 1)], \
            post_gain_speech[slice(segment_first_sample, segment_last_sample + 1)]
        gain_segment = np.clip(divide(post_gain_speech_segment, pre_gain_speech_segment), 0, 1)
        noisy_residual_echo_segment = subtract(pre_gain_speech_segment, speech_reference_segment)
        gain_over_speech_segment = multiply(gain_segment, speech_reference_segment)
        gain_over_noisy_residual_echo_segment = multiply(gain_segment, noisy_residual_echo_segment)

        if compensate:
            compensation_factor_segment_speech_reference = divide(
                sum(multiply(gain_over_speech_segment, speech_reference_segment)),
                add(pow(np.linalg.norm(speech_reference_segment), 2), lambda_reg))
            speech_reference_segment = \
                multiply(compensation_factor_segment_speech_reference, speech_reference_segment)

        current_resl = \
            20 * np.log10(divide(np.linalg.norm(add(noisy_residual_echo_segment, lambda_reg)),
                                 np.linalg.norm(add(gain_over_noisy_residual_echo_segment, lambda_reg))))
        current_dsml = \
            20 * np.log10(divide(np.linalg.norm(add(speech_reference_segment, lambda_reg)),
                                 np.linalg.norm(add(np.subtract(speech_reference_segment, gain_over_speech_segment),
                                                    lambda_reg))))
        resl.append(current_resl)
        dsml.append(current_dsml)
        segment_first_sample += hop_size_samples
        segment_last_sample += hop_size_samples
    resl = np.asarray(resl)
    dsml = np.asarray(dsml)

    return resl, dsml
