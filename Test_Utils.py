"""
@File: Test_Utils
@Time: 3/18/25 12:41 AM
@Author: liincuan
@Description: 
"""

import unittest
import os
import numpy as np
from scipy.io import wavfile
import tempfile
import shutil
from utils import parse_files_from_path, read_wav_files, save_and_plot


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sub_dir = os.path.join(self.temp_dir, "test_batch")
        os.makedirs(self.sub_dir)
        self.fs = 16000
        self.duration = 1.0
        self.samples = int(self.fs * self.duration)
        self.data = np.random.normal(0, 1, self.samples).astype(np.int16)  # 随机信号

        self.patterns = ('reference', 'pre_gain', 'post_gain')
        self.wav_files = [
            os.path.join(self.sub_dir, f"signal_{p}.wav") for p in self.patterns
        ]


        for wav_file in self.wav_files:
            wavfile.write(wav_file, self.fs, self.data)

        self.log_file = os.path.join(self.temp_dir, "log.txt")
        with open(self.log_file, 'w') as f:
            f.write("Initial log content\n")

    def tearDown(self):

        shutil.rmtree(self.temp_dir)

    def test_parse_files_from_path(self):

        paths_list = parse_files_from_path(self.temp_dir, self.patterns)


        self.assertEqual(len(paths_list), 1)
        self.assertEqual(len(paths_list[0]), 3)
        for path, pattern in zip(paths_list[0], self.patterns):
            self.assertIn(pattern, path)
            self.assertTrue(os.path.exists(path))

        bad_patterns = ('ref', 'pre', 'wrong')
        result = parse_files_from_path(self.temp_dir, bad_patterns)
        self.assertIsInstance(result, str)
        self.assertIn("Error", result)

    def test_read_wav_files(self):

        paths_list = parse_files_from_path(self.temp_dir, self.patterns)[0]
        wav_array, fs = read_wav_files(paths_list, self.patterns, plot=False)

        self.assertEqual(fs, self.fs)
        self.assertEqual(wav_array.shape, (self.samples, 3))
        self.assertTrue(np.allclose(wav_array[:, 0], self.data / 2 ** 15, atol=1e-5))

        bad_file = os.path.join(self.sub_dir, "bad_ref.wav")
        wavfile.write(bad_file, 8000, self.data)
        bad_paths = [bad_file] + paths_list[1:]
        with self.assertRaises(SystemExit):
            read_wav_files(bad_paths, self.patterns)

    def test_save_and_plot(self):

        resl = np.array([10.0, 20.0, 15.0])
        dsml = np.array([5.0, 8.0, 6.0])
        paths_list = parse_files_from_path(self.temp_dir, self.patterns)[0]


        with open(self.log_file, 'a') as log_handle:
            updated_log = save_and_plot(resl, dsml, paths_list, log_handle, save_to_text=True, plot=False)


        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertIn("RESL: 15.00 ± 4.08 dB", content)
        self.assertIn("DSML: 6.33 ± 1.25 dB", content)
        self.assertIn(self.sub_dir, content)


if __name__ == '__main__':
    unittest.main()