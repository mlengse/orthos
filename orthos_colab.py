# -*- coding: utf-8 -*-
"""orthos_colab.py

Original Javascript Port of Frank Liang's PatGen algorithm to Python/CUDA.
Designed for Google Colab (T4/A100 GPU).

Usage:
    python orthos_colab.py dictionary.txt patterns_in.txt patterns_out.txt
"""

import sys
import os
import time
import argparse
import subprocess
import re

# ==========================================
# AUTO-INSTALL DEPENDENCIES (COLAB)
# ==========================================
def install_dependencies():
    """Installs necessary packages if running in Colab or missing."""
    packages = ["cupy-cuda12x", "numba", "numpy"]

    # Check if we are in Colab
    in_colab = "google.colab" in sys.modules

    if in_colab:
        print("⚡ Google Colab detected. checking dependencies...")
        try:
            import cupy
            import numba
            import numpy
            print("   Dependencies already installed.")
        except ImportError:
            print("   Installing CuPy, Numba, and NumPy...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
            print("   Installation complete.")
    else:
        # Local environment check
        try:
            import cupy
            import numba
            import numpy
        except ImportError:
            print("⚠️  Warning: CuPy or Numba not found. GPU acceleration will not work.")
            print("   Please install them: pip install numpy numba cupy-cuda12x")

install_dependencies()

# ==========================================
# IMPORTS
# ==========================================
import numpy as np

try:
    from numba import jit, prange, int32, uint32, float32, uint8
except ImportError:
    print("⚠️  Warning: Numba not found. CPU optimization will be disabled.")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    int32 = int
    uint32 = int
    float32 = float
    uint8 = int

try:
    import cupy as cp
    from numba import cuda
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    cuda = None
    GPU_AVAILABLE = False

# ==========================================
# CONSTANTS
# ==========================================
MAX_VAL = 10
MAX_DOT = 15
MAX_LEN = 50
TRIE_SIZE = 55000 * 500
MAX_OPS = 510 * 100
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

# Character Classes
DIGIT_CLASS = 1
HYF_CLASS = 2
LETTER_CLASS = 3
NO_HYF = 10     # ''
ERR_HYF = 11    # '#'
IS_HYF = 12     # '-'
FOUND_HYF = 13  # '*'


# ==========================================
# GPU KERNELS
# ==========================================
if GPU_AVAILABLE:
    @cuda.jit
    def kernel_hyphenate(words, word_lens, trie_c, trie_l, trie_r, ops_val, ops_dot, ops_op, trie_root, hvals, no_more, pat_len, pat_dot, hyph_level, max_val):
        idx = cuda.grid(1)
        if idx >= words.shape[0]:
            return

        # BOLT OPTIMIZATION: Use Local Memory for accumulation
        # Buffering hvals/no_more in registers/L1 cache reduces global memory traffic
        # and allows for coalesced burst writes at the end.
        l_hvals = cuda.local.array(MAX_LEN, uint8)
        l_no_more = cuda.local.array(MAX_LEN, uint8)

        # Initialize local buffers
        for i in range(MAX_LEN):
            l_hvals[i] = 0
            l_no_more[i] = 0

        wlen = word_lens[idx]

        # Iterate backwards through the word
        for spos in range(wlen - 1, -1, -1):
            fpos = spos + 1
            if fpos >= MAX_LEN: continue
            if fpos >= wlen: continue

            char_code = words[idx, fpos]
            t = int32(trie_root) + int32(char_code)

            # Traverse Trie
            while True:
                # Bound check for t
                if t >= trie_c.shape[0]: break

                if trie_c[t] == char_code:
                    h = trie_r[t]
                    while h > 0:
                        op_val = ops_val[h]
                        op_dot = ops_dot[h]
                        op_next = ops_op[h]

                        dpos = spos + op_dot

                        if dpos < MAX_LEN:
                            # Update hvals
                            if op_val < max_val:
                                if l_hvals[dpos] < op_val:
                                    l_hvals[dpos] = op_val

                            # Update no_more
                            if op_val >= hyph_level:
                                cond1 = (fpos - pat_len) <= (dpos - pat_dot)
                                cond2 = (dpos - pat_dot) <= spos
                                if cond1 and cond2:
                                    l_no_more[dpos] = 1

                        h = op_next

                    t = int32(trie_l[t])
                    if t == 0:
                        break

                    fpos += 1
                    if fpos >= wlen:
                        break
                    char_code = words[idx, fpos]
                    t = int32(t) + int32(char_code)
                else:
                    break

        # Flush local buffers to global memory (Coalesced Write)
        for i in range(MAX_LEN):
            # Only write if non-zero to save bandwidth?
            # Actually, we need to overwrite the output buffer (which was reset to 0).
            # Writing 0 is fine.
            # hvals is (N, MAX_LEN) F-order -> hvals[idx, i] = hvals[idx + i*N]
            # Adjacent threads (idx, idx+1) write to adjacent addresses. Perfect.
            hvals[idx, i] = l_hvals[i]
            no_more[idx, i] = l_no_more[i]

    @cuda.jit
    def kernel_update_dots(dots, hvals, dotw, word_lens, hyf_min, hyf_max,
                           found_hyf, err_hyf, is_hyf,
                           good_count, bad_count, miss_count):
        """
        Equivalent to change_dots() in orthos.js.
        Updates the 'dots' array based on calculated 'hvals'.
        """
        idx = cuda.grid(1)
        if idx >= dots.shape[0]:
            return

        wlen = word_lens[idx]
        dpos = wlen - hyf_max

        while dpos >= hyf_min:
            # If hval is odd, it means we found a hyphen here
            if (hvals[idx, dpos] % 2) == 1:
                dots[idx, dpos] += 1
            dpos -= 1

    @cuda.jit
    def kernel_extract_candidates(words, word_lens, dots, dotw, no_more,
                                  pat_len, pat_dot, good_dot, bad_dot,
                                  dot_min, dot_max,
                                  out_patterns, out_weights, out_mask):
        idx = cuda.grid(1)
        if idx >= words.shape[0]:
            return

        wlen = word_lens[idx]

        # dpos loop: wlen - dot_max down to dot_min
        for dpos in range(wlen - dot_max, dot_min - 1, -1):
            if no_more[idx, dpos] == 0:
                current_dot = dots[idx, dpos]
                is_good = (current_dot == good_dot)
                is_bad = (current_dot == bad_dot)

                if is_good or is_bad:
                    spos = dpos - pat_dot
                    # out_idx = idx * MAX_LEN + dpos (REMOVED: Use coalesced indexing [idx, dpos, k])

                    # Check bounds
                    if spos + 1 >= 0 and spos + 1 + pat_len <= wlen:
                        for k in range(pat_len):
                            out_patterns[idx, dpos, k] = words[idx, spos + 1 + k]

                        weight = dotw[idx, dpos]

                        if is_good:
                            out_weights[idx, dpos, 0] = weight
                            out_weights[idx, dpos, 1] = 0.0
                        else:
                            out_weights[idx, dpos, 0] = 0.0
                            out_weights[idx, dpos, 1] = float(weight)

                        out_mask[idx, dpos] = 1

# ==========================================
# CPU KERNELS (Numba Optimized)
# ==========================================
@jit(nopython=True, parallel=True)
def kernel_hyphenate_cpu(words, word_lens, trie_c, trie_l, trie_r, ops_val, ops_dot, ops_op, trie_root, hvals, no_more, pat_len, pat_dot, hyph_level, max_val):
    n_words = words.shape[0]
    for idx in prange(n_words):
        wlen = word_lens[idx]

        # Iterate backwards through the word
        for spos in range(wlen - 1, -1, -1):
            fpos = spos + 1
            if fpos >= MAX_LEN: continue
            if fpos >= wlen: continue

            char_code = words[idx, fpos]
            t = int32(trie_root) + int32(char_code)

            # Traverse Trie
            while True:
                # Bound check for t
                if t >= trie_c.shape[0]: break

                if trie_c[t] == char_code:
                    h = trie_r[t]
                    while h > 0:
                        op_val = ops_val[h]
                        op_dot = ops_dot[h]
                        op_next = ops_op[h]

                        dpos = spos + op_dot

                        if dpos < MAX_LEN:
                            # Update hvals
                            if op_val < max_val:
                                if hvals[idx, dpos] < op_val:
                                    hvals[idx, dpos] = op_val

                            # Update no_more
                            if op_val >= hyph_level:
                                cond1 = (fpos - pat_len) <= (dpos - pat_dot)
                                cond2 = (dpos - pat_dot) <= spos
                                if cond1 and cond2:
                                    no_more[idx, dpos] = 1

                        h = op_next

                    t = int32(trie_l[t])
                    if t == 0:
                        break

                    fpos += 1
                    if fpos >= wlen:
                        break
                    char_code = words[idx, fpos]
                    t = int32(t) + int32(char_code)
                else:
                    break

@jit(nopython=True, parallel=True)
def kernel_update_dots_cpu(dots, hvals, dotw, word_lens, hyf_min, hyf_max):
    n_words = dots.shape[0]
    for idx in prange(n_words):
        wlen = word_lens[idx]
        dpos = wlen - hyf_max

        while dpos >= hyf_min:
            # If hval is odd, it means we found a hyphen here
            if (hvals[idx, dpos] % 2) == 1:
                dots[idx, dpos] += 1
            dpos -= 1

@jit(nopython=True, parallel=True)
def kernel_extract_candidates_cpu(words, word_lens, dots, dotw, no_more,
                              pat_len, pat_dot, good_dot, bad_dot,
                              dot_min, dot_max,
                              out_patterns, out_weights, out_mask):
    n_words = words.shape[0]
    for idx in prange(n_words):
        wlen = word_lens[idx]

        # dpos loop: wlen - dot_max down to dot_min
        for dpos in range(wlen - dot_max, dot_min - 1, -1):
            if no_more[idx, dpos] == 0:
                current_dot = dots[idx, dpos]
                is_good = (current_dot == good_dot)
                is_bad = (current_dot == bad_dot)

                if is_good or is_bad:
                    spos = dpos - pat_dot
                    # out_idx = idx * MAX_LEN + dpos (REMOVED)

                    # Check bounds
                    if spos + 1 >= 0 and spos + 1 + pat_len <= wlen:
                        for k in range(pat_len):
                            out_patterns[idx, dpos, k] = words[idx, spos + 1 + k]

                        weight = dotw[idx, dpos]

                        if is_good:
                            out_weights[idx, dpos, 0] = weight
                            out_weights[idx, dpos, 1] = 0.0
                        else:
                            out_weights[idx, dpos, 0] = 0.0
                            out_weights[idx, dpos, 1] = float(weight)

                        out_mask[idx, dpos] = 1

# ==========================================
# ORTHOS ENGINE
# ==========================================
class OrthosEngine:
    def __init__(self):
        self.xint = {}
        self.xext = []
        self.xclass = {}
        
        # Dynamic trie sizing
        self.trie_size = TRIE_SIZE
        self.max_ops = MAX_OPS
        
        # Optimization: Use int32 for Trie arrays to save 50% memory (27.5M ints)
        # and avoid casting during GPU transfer.
        self.trie_c = np.zeros(self.trie_size, dtype=np.int32)
        self.trie_l = np.zeros(self.trie_size, dtype=np.int32)
        self.trie_r = np.zeros(self.trie_size, dtype=np.int32)
        self.trie_taken = np.zeros(self.trie_size, dtype=np.uint8)

        self.trie_dirty = True

        # Structure of Arrays (SoA) for OPS to avoid Python loop overhead in sync
        self.ops_val = np.zeros(self.max_ops + 1, dtype=np.uint8)
        self.ops_dot = np.zeros(self.max_ops + 1, dtype=np.uint8)
        self.ops_op = np.zeros(self.max_ops + 1, dtype=np.uint32)
        # self.ops = [{'val':0, 'dot':0, 'op':0} for _ in range(self.max_ops + 1)] # 1-based (REMOVED)

        # Trie packing structures
        self.trieq_c = np.zeros(50, dtype=np.int64)
        self.trieq_l = np.zeros(50, dtype=np.int64)
        self.trieq_r = np.zeros(50, dtype=np.int64)
        self.qmax = 0
        self.qmax_thresh = 5
        self.trie_max = 0
        self.trie_bmax = 0
        self.op_count = 0

        self.trie_root = 1

        self.initialize_chars()
        self.init_pattern_trie()

        # Data
        self.words = None
        self.word_lens = None
        self.dots = None
        self.dotw = None

        # GPU Cache
        self.d_words = None
        self.d_word_lens = None
        self.d_dots = None
        self.d_dotw = None

        # GPU Workspace Buffers (Pre-allocated)
        self.d_hvals = None
        self.d_no_more = None
        self.d_dots_working = None
        self.d_out_mask = None

        self.gpu_available = False
        if GPU_AVAILABLE:
            try:
                cuda.get_current_device()
                self.gpu_available = True
                print(f"✅ GPU Detected: {cuda.get_current_device().name.decode('utf-8')}")
            except Exception as e:
                print(f"⚠️ GPU Import successful but device not ready: {e}")

    def initialize_chars(self):
        digits = "0123456789"
        for i, c in enumerate(digits):
            code = ord(c)
            self.xext.append(c)
            self.xint[code] = i
            self.xclass[code] = DIGIT_CLASS

        # Special chars
        specials = {
            '.': (10, LETTER_CLASS), # edge_of_word
            '#': (11, HYF_CLASS),    # err_hyf
            '-': (12, HYF_CLASS),    # is_hyf
            '*': (13, HYF_CLASS)     # found_hyf
        }

        while len(self.xext) <= 13:
            self.xext.append('')

        for char, (internal, cls) in specials.items():
            code = ord(char)
            self.xext[internal] = char
            self.xint[code] = internal
            self.xclass[code] = cls

        self.cmin = 1
        self.cmax = len(self.xext) - 1

    def init_pattern_trie(self):
        c = 0
        while c <= self.cmax:
            idx = self.trie_root + c
            self.trie_c[idx] = c
            self.trie_l[idx] = 0
            self.trie_r[idx] = 0
            self.trie_taken[idx] = 0
            c += 1
        self.trie_taken[self.trie_root] = 1
        self.trie_bmax = self.trie_root
        self.trie_max = self.trie_root + self.cmax

        self.trie_l[0] = self.trie_max + 1
        self.trie_r[self.trie_max + 1] = 0
        self.trie_dirty = True

    def sync_trie_to_gpu(self):
        if not self.gpu_available: return
        if not self.trie_dirty: return

        limit = self.trie_max + 1

        # Optimization: Remove astype(int32) since arrays are already int32
        self.d_trie_c = cp.asarray(self.trie_c[:limit])
        self.d_trie_l = cp.asarray(self.trie_l[:limit])
        self.d_trie_r = cp.asarray(self.trie_r[:limit])

        self.trie_dirty = False

        # Optimized Transfer: Direct array copy (No Python loop)
        self.d_ops_val = cp.asarray(self.ops_val)
        self.d_ops_dot = cp.asarray(self.ops_dot)
        self.d_ops_op = cp.asarray(self.ops_op)

    def sync_dictionary_to_gpu(self):
        if not self.gpu_available: return

        if (self.d_words is not None): return

        print("   [GPU] Uploading dictionary to VRAM...")
        self.d_words = cp.asarray(self.words, order='F')
        self.d_word_lens = cp.asarray(self.word_lens)
        self.d_dots = cp.asarray(self.dots, order='F')
        self.d_dotw = cp.asarray(self.dotw, order='F')

        # BOLT OPTIMIZATION: Pre-allocate workspace buffers
        # Avoids repeating malloc/free in the inner generation loop
        print("   [GPU] Allocating workspace buffers...")
        n_words = self.words.shape[0]
        self.d_hvals = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')
        self.d_no_more = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')
        # d_dots_working must be same shape/type as d_dots
        self.d_dots_working = cp.zeros_like(self.d_dots)
        # d_out_mask is also constant size
        self.d_out_mask = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')

    def unpack(self, s):
        qmax = 1
        c = self.cmin
        while c <= self.cmax:
            t = s + c
            if self.trie_c[t] == c:
                self.trieq_c[qmax] = c
                self.trieq_l[qmax] = self.trie_l[t]
                self.trieq_r[qmax] = self.trie_r[t]
                qmax += 1
                self.trie_r[self.trie_l[0]] = t
                self.trie_l[t] = self.trie_l[0]
                self.trie_l[0] = t
                self.trie_r[t] = 0
                self.trie_c[t] = 0
            c += 1
        self.trie_taken[s] = 0
        self.qmax = qmax - 1

    def first_fit(self):
        t = self.trie_r[self.trie_max + 1] if self.qmax > self.qmax_thresh else 0
        while True:
            t = int(self.trie_l[t])
            s = int(t) - int(self.trieq_c[1])
            if s < 0 or s > self.trie_size - len(self.xext):
                raise RuntimeError("Trie Overflow")

            while self.trie_bmax < s:
                self.trie_bmax += 1
                self.trie_taken[self.trie_bmax] = 0
                idx = self.trie_bmax + self.cmax
                self.trie_c[idx] = 0
                self.trie_l[idx] = self.trie_bmax + len(self.xext)
                self.trie_r[self.trie_bmax + len(self.xext)] = idx

            if self.trie_taken[s] == 0:
                q = self.qmax
                collision = False
                while q >= 2:
                    if self.trie_c[s + self.trieq_c[q]] != 0:
                        collision = True
                        break
                    q -= 1
                if not collision:
                    break

        for q in range(1, self.qmax + 1):
            t_node = s + self.trieq_c[q]
            self.trie_l[self.trie_r[t_node]] = self.trie_l[t_node]
            self.trie_r[self.trie_l[t_node]] = self.trie_r[t_node]
            self.trie_c[t_node] = self.trieq_c[q]
            self.trie_l[t_node] = self.trieq_l[q]
            self.trie_r[t_node] = self.trieq_r[q]
            if t_node > self.trie_max:
                self.trie_max = t_node

        self.trie_taken[s] = 1
        return s

    def new_trie_op(self, val, dot, next_op):
        h = ((int(next_op) + 313 * int(dot) + 361 * int(val)) % MAX_OPS) + 1
        while True:
            if self.ops_val[h] == 0:
                self.op_count += 1
                if self.op_count == MAX_OPS:
                    raise RuntimeError("Outputs Overflow")
                self.ops_val[h] = val
                self.ops_dot[h] = dot
                self.ops_op[h] = next_op
                return h
            if (self.ops_val[h] == val and self.ops_dot[h] == dot and self.ops_op[h] == next_op):
                return h
            if h > 1:
                h -= 1
            else:
                h = MAX_OPS

    def insert_pattern(self, pat, val, dot):
        pat_len = len(pat)
        i = 0
        s = self.trie_root + pat[i]
        t = self.trie_l[s]

        while t > 0 and i < pat_len - 1:
            i += 1
            t += pat[i]
            if self.trie_c[t] != pat[i]:
                if self.trie_c[t] == 0:
                    self.trie_l[self.trie_r[t]] = self.trie_l[t]
                    self.trie_r[self.trie_l[t]] = self.trie_r[t]
                    self.trie_c[t] = pat[i]
                    self.trie_l[t] = 0
                    self.trie_r[t] = 0
                    if t > self.trie_max:
                        self.trie_max = t
                else:
                    self.unpack(t - pat[i])
                    self.trieq_c[self.qmax + 1] = pat[i]
                    self.trieq_l[self.qmax + 1] = 0
                    self.trieq_r[self.qmax + 1] = 0
                    self.qmax += 1
                    t = self.first_fit()
                    self.trie_l[s] = t
                    t += pat[i]
            s = t
            t = self.trie_l[s]

        self.trieq_l[1] = 0
        self.trieq_r[1] = 0
        self.qmax = 1

        while i < pat_len - 1:
            i += 1
            self.trieq_c[1] = pat[i]
            t = self.first_fit()
            self.trie_l[s] = t
            s = t + pat[i]

        self.trie_r[s] = self.new_trie_op(val, dot, self.trie_r[s])
        self.trie_dirty = True

    def validate_file(self, filepath):
        """Validates that a file is safe to read (exists, is file, size limit)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        if not os.path.isfile(filepath):
            raise ValueError(f"Path is not a regular file: {filepath}")

        size = os.path.getsize(filepath)
        if size > MAX_FILE_SIZE:
            raise ValueError(f"File {filepath} (size: {size} bytes) exceeds maximum allowed size of {MAX_FILE_SIZE} bytes.")

    def validate_output_file(self, filepath):
        """
        Validates that the output path is safe to write to.
        - Must not be a directory.
        - Parent directory must exist.
        - If file exists, must be a regular file.
        """
        if os.path.isdir(filepath):
            raise IsADirectoryError(f"Output path is a directory: {filepath}")

        parent = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(parent):
            raise FileNotFoundError(f"Parent directory does not exist: {parent}")
        if not os.path.isdir(parent):
            raise NotADirectoryError(f"Parent path is not a directory: {parent}")

        if os.path.exists(filepath) and not os.path.isfile(filepath):
             raise ValueError(f"Output path exists but is not a regular file: {filepath}")

    def load_dictionary(self, filepath):
        print(f"📖 Loading dictionary from {filepath}...")
        self.validate_file(filepath)
        words_list = []
        dots_list = []
        dotw_list = []
        lens_list = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                word_ints = [self.xint[ord('.')]]
                dots = [NO_HYF]
                dotw = [1]
                current_weight = 1

                i = 0
                while i < len(line):
                    char = line[i]
                    code = ord(char)
                    if code not in self.xint:
                        self.xext.append(char)
                        idx = len(self.xext) - 1
                        self.xint[code] = idx
                        self.xclass[code] = LETTER_CLASS
                        self.cmax = len(self.xext) - 1

                    cls = self.xclass[code]
                    if cls == LETTER_CLASS:
                        word_ints.append(self.xint[code])
                        dots.append(NO_HYF)
                        dotw.append(current_weight)
                    elif cls == HYF_CLASS:
                        dots[-1] = self.xint[code]
                    elif cls == DIGIT_CLASS:
                        digit_val = self.xint[code]
                        if len(word_ints) == 1:
                            current_weight = digit_val
                        else:
                            dotw[-1] = digit_val
                    i += 1

                word_ints.append(self.xint[ord('.')])
                dots.append(NO_HYF)
                dotw.append(current_weight)

                # Check Length
                if len(word_ints) > MAX_LEN:
                    word_ints = word_ints[:MAX_LEN]
                    dots = dots[:MAX_LEN]
                    dotw = dotw[:MAX_LEN]

                lens_list.append(len(word_ints))

                # Pad
                pad = MAX_LEN - len(word_ints)
                word_ints.extend([0]*pad)
                dots.extend([0]*pad)
                dotw.extend([0]*pad)

                words_list.append(word_ints)
                dots_list.append(dots)
                dotw_list.append(dotw)

        # GPU OPTIMIZATION: Sort by word length
        # This reduces warp divergence in the GPU kernel because threads in a block
        # will process words of similar length and finish at similar times.
        if words_list:
            combined = list(zip(words_list, dots_list, dotw_list, lens_list))
            combined.sort(key=lambda x: x[3]) # Sort by length
            words_list, dots_list, dotw_list, lens_list = zip(*combined)

        self.words = np.array(words_list, dtype=np.uint32, order='F')
        self.dots = np.array(dots_list, dtype=np.uint32, order='F')
        self.dotw = np.array(dotw_list, dtype=np.uint32, order='F')
        self.word_lens = np.array(lens_list, dtype=np.int32)

        # Reset Cache
        self.d_words = None
        self.d_hvals = None
        self.d_no_more = None
        self.d_dots_working = None
        self.d_out_mask = None

        # Reset Trie (new chars might have been added)
        self.trie_c = np.zeros(TRIE_SIZE, dtype=np.int32)
        self.trie_l = np.zeros(TRIE_SIZE, dtype=np.int32)
        self.trie_r = np.zeros(TRIE_SIZE, dtype=np.int32)
        self.trie_taken = np.zeros(TRIE_SIZE, dtype=np.uint8)

        self.ops_val[:] = 0
        self.ops_dot[:] = 0
        self.ops_op[:] = 0

        self.op_count = 0
        self.trie_max = 0
        self.trie_bmax = 0
        self.init_pattern_trie()

        print(f"   Loaded {len(words_list)} words (Sorted by length).")

    def load_patterns(self, filepath):
        print(f"📖 Loading patterns from {filepath}...")
        if not os.path.exists(filepath):
            print(f"   File not found. Starting with empty patterns.")
            return

        self.validate_file(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse TeX patterns
        tokens = re.findall(r'[^\s{}]+', content)
        count = 0
        for token in tokens:
            if '\\patterns' in token or token == '}': continue
            if '%' in token: continue

            # Parse 1ba -> val=1 at 0, b, a
            # logic from orthos_gpu
            letters = []
            values = {}
            letter_pos = 0

            i = 0
            while i < len(token):
                c = token[i]
                if c.isdigit():
                    values[letter_pos] = int(c)
                else:
                    letters.append(c)
                    letter_pos += 1
                i += 1

            if not letters: continue

            # Encode
            pat_ints = []
            for ch in letters:
                code = ord(ch)
                if code not in self.xint:
                     self.xext.append(ch)
                     self.xint[code] = len(self.xext) - 1
                     self.xclass[code] = LETTER_CLASS
                     self.cmax = len(self.xext) - 1
                pat_ints.append(self.xint[code])
                
            # Insert
            for dot_pos, val in values.items():
                if val > 0:
                    try:
                        self.insert_pattern(pat_ints, val, dot_pos)
                        count += 1
                    except RuntimeError:
                        pass
        print(f"   Loaded {count} pattern points.")

    def delete_patterns(self, s):
        c = self.cmin
        all_freed = True
        while c <= self.cmax:
            t = s + c
            if self.trie_c[t] == c:
                ops_list = []
                h = int(self.trie_r[t])
                while h > 0:
                    ops_list.append((self.ops_val[h], self.ops_dot[h], self.ops_op[h]))
                    h = self.ops_op[h]
                new_head = 0
                for val, dot, _ in reversed(ops_list):
                    if val != MAX_VAL and val > 0:
                        new_head = self.new_trie_op(val, dot, new_head)
                self.trie_r[t] = new_head
                if self.trie_l[t] > 0:
                    self.trie_l[t] = self.delete_patterns(self.trie_l[t])
                if self.trie_l[t] > 0 or self.trie_r[t] > 0 or s == self.trie_root:
                    all_freed = False
                else:
                    self.trie_l[self.trie_r[self.trie_max + 1]] = t
                    self.trie_r[t] = self.trie_r[self.trie_max + 1]
                    self.trie_l[t] = self.trie_max + 1
                    self.trie_r[self.trie_max + 1] = t
                    self.trie_c[t] = 0
            c += 1
        if all_freed:
            self.trie_taken[s] = 0
            s = 0
        self.trie_dirty = True
        return s

    def delete_bad_patterns(self):
        self.delete_patterns(self.trie_root)
        self.trie_dirty = True
        for h in range(1, MAX_OPS + 1):
            if self.ops_val[h] == MAX_VAL:
                self.ops_val[h] = 0
        self.qmax_thresh = 7

    def do_dictionary_gpu(self, pat_len, pat_dot, hyph_level, good_wt, bad_wt, thresh, left_hyphen_min, right_hyphen_min):
        if not self.gpu_available:
            print("⚠️ GPU not available. Switching to CPU.")
            return self.do_dictionary_cpu(pat_len, pat_dot, hyph_level, good_wt, bad_wt, thresh, left_hyphen_min, right_hyphen_min)

        self.sync_trie_to_gpu()
        self.sync_dictionary_to_gpu()

        n_words = self.words.shape[0]

        # Reuse pre-allocated buffers
        # Reset them to 0 or initial state
        self.d_hvals.fill(0)
        self.d_no_more.fill(0)

        # Reset dots working copy efficiently
        cp.copyto(self.d_dots_working, self.d_dots)

        threadsperblock = 128
        blockspergrid = (n_words + (threadsperblock - 1)) // threadsperblock

        # 1. Hyphenate
        kernel_hyphenate[blockspergrid, threadsperblock](
            self.d_words, self.d_word_lens,
            self.d_trie_c, self.d_trie_l, self.d_trie_r,
            self.d_ops_val, self.d_ops_dot, self.d_ops_op,
            self.trie_root,
            self.d_hvals, self.d_no_more,
            pat_len, pat_dot, hyph_level, MAX_VAL
        )

        hyf_min = left_hyphen_min + 1
        hyf_max = right_hyphen_min + 1

        # 2. Update Dots
        d_good_count = cp.array([0], dtype=cp.uint32)
        d_bad_count = cp.array([0], dtype=cp.uint32)
        d_miss_count = cp.array([0], dtype=cp.uint32)

        kernel_update_dots[blockspergrid, threadsperblock](
            self.d_dots_working, self.d_hvals, self.d_dotw, self.d_word_lens,
            hyf_min, hyf_max,
            FOUND_HYF, ERR_HYF, IS_HYF,
            d_good_count, d_bad_count, d_miss_count
        )

        if (hyph_level % 2) == 1:
            good_dot = IS_HYF
            bad_dot = NO_HYF
        else:
            good_dot = ERR_HYF
            bad_dot = FOUND_HYF

        dot_min = pat_dot
        if dot_min < (left_hyphen_min + 1): dot_min = left_hyphen_min + 1
        dot_max = pat_len - pat_dot
        if dot_max < (right_hyphen_min + 1): dot_max = right_hyphen_min + 1

        # BOLT OPTIMIZATION: Use Coalesced Memory Layout (order='F')
        # Structure: (n_words, MAX_LEN, pat_len)
        # This ensures threads accessing the same 'k' and 'dpos' but different 'idx' (word index)
        # access adjacent memory locations, maximizing memory bandwidth.
        d_out_patterns = cp.zeros((n_words, MAX_LEN, pat_len), dtype=cp.uint32, order='F')
        d_out_weights = cp.zeros((n_words, MAX_LEN, 2), dtype=cp.float32, order='F')

        # Reuse pre-allocated mask (reset to 0)
        self.d_out_mask.fill(0)

        # 3. Extract Candidates
        kernel_extract_candidates[blockspergrid, threadsperblock](
            self.d_words, self.d_word_lens, self.d_dots_working, self.d_dotw, self.d_no_more,
            pat_len, pat_dot, good_dot, bad_dot,
            dot_min, dot_max,
            d_out_patterns, d_out_weights, self.d_out_mask
        )

        # Aggregation
        mask = self.d_out_mask.astype(bool)
        valid_patterns = d_out_patterns[mask]
        valid_weights = d_out_weights[mask]

        if valid_patterns.shape[0] == 0:
            return 0, 0

        # Lexsort
        keys = cp.vstack([valid_patterns[:, i] for i in range(pat_len - 1, -1, -1)])
        sorted_indices = cp.lexsort(keys)

        sorted_patterns = valid_patterns[sorted_indices]
        sorted_weights = valid_weights[sorted_indices]

        diff = sorted_patterns[1:] != sorted_patterns[:-1]
        boundaries = cp.any(diff, axis=1)

        run_ids = cp.zeros(len(sorted_patterns), dtype=cp.int32)
        run_ids[1:] = cp.cumsum(boundaries)

        num_unique = int(run_ids[-1]) + 1
        unique_weights = cp.zeros((num_unique, 2), dtype=cp.float32)
        cp.add.at(unique_weights, run_ids, sorted_weights)

        change_points = cp.flatnonzero(boundaries) + 1
        unique_indices = cp.concatenate((cp.array([0]), change_points))
        unique_patterns_arr = sorted_patterns[unique_indices]

        # Transfer back to CPU
        cpu_weights = unique_weights.get()
        cpu_patterns = unique_patterns_arr.get()

        good_pat_count = 0
        bad_pat_count = 0

        for i in range(num_unique):
            w_good = cpu_weights[i, 0]
            w_bad = cpu_weights[i, 1]
            pat = cpu_patterns[i].tolist()
            try:
                if (good_wt * w_good) < thresh:
                    self.insert_pattern(pat, MAX_VAL, pat_dot)
                    bad_pat_count += 1
                elif (good_wt * w_good - bad_wt * w_bad) >= thresh:
                    self.insert_pattern(pat, hyph_level, pat_dot)
                    good_pat_count += 1
            except RuntimeError:
                pass
        
        return good_pat_count, bad_pat_count

    def do_dictionary_cpu(self, pat_len, pat_dot, hyph_level, good_wt, bad_wt, thresh, left_hyphen_min, right_hyphen_min):
        """CPU Fallback implementation using Numba"""
        print("   [CPU] Running processing on CPU...")

        n_words = self.words.shape[0]
        hvals = np.zeros((n_words, MAX_LEN), dtype=np.uint8, order='F')
        no_more = np.zeros((n_words, MAX_LEN), dtype=np.uint8, order='F')

        # Working copy of dots
        dots_working = np.copy(self.dots)

        # 1. Hyphenate
        kernel_hyphenate_cpu(
            self.words, self.word_lens,
            self.trie_c, self.trie_l, self.trie_r,
            self.ops_val, self.ops_dot, self.ops_op,
            self.trie_root,
            hvals, no_more,
            pat_len, pat_dot, hyph_level, MAX_VAL
        )

        hyf_min = left_hyphen_min + 1
        hyf_max = right_hyphen_min + 1

        # 2. Update Dots
        kernel_update_dots_cpu(
            dots_working, hvals, self.dotw, self.word_lens,
            hyf_min, hyf_max
        )

        if (hyph_level % 2) == 1:
            good_dot = IS_HYF
            bad_dot = NO_HYF
        else:
            good_dot = ERR_HYF
            bad_dot = FOUND_HYF

        dot_min = pat_dot
        if dot_min < (left_hyphen_min + 1): dot_min = left_hyphen_min + 1
        dot_max = pat_len - pat_dot
        if dot_max < (right_hyphen_min + 1): dot_max = right_hyphen_min + 1

        # Pre-allocate output arrays
        # Updated to match GPU layout (order='F') for consistency
        out_patterns = np.zeros((n_words, MAX_LEN, pat_len), dtype=np.uint32, order='F')
        out_weights = np.zeros((n_words, MAX_LEN, 2), dtype=np.float32, order='F')
        out_mask = np.zeros((n_words, MAX_LEN), dtype=np.uint8, order='F')

        # 3. Extract Candidates
        kernel_extract_candidates_cpu(
            self.words, self.word_lens, dots_working, self.dotw, no_more,
            pat_len, pat_dot, good_dot, bad_dot,
            dot_min, dot_max,
            out_patterns, out_weights, out_mask
        )

        # Aggregation
        mask = out_mask.astype(bool)
        valid_patterns = out_patterns[mask]
        valid_weights = out_weights[mask]

        if valid_patterns.shape[0] == 0:
            return 0, 0

        # Lexsort (NumPy)
        # np.lexsort takes keys in reverse order of significance
        keys = tuple(valid_patterns[:, i] for i in range(pat_len - 1, -1, -1))
        sorted_indices = np.lexsort(keys)

        sorted_patterns = valid_patterns[sorted_indices]
        sorted_weights = valid_weights[sorted_indices]

        # Find boundaries
        diff = sorted_patterns[1:] != sorted_patterns[:-1]
        boundaries = np.any(diff, axis=1)

        run_ids = np.zeros(len(sorted_patterns), dtype=np.int32)
        run_ids[1:] = np.cumsum(boundaries)

        num_unique = int(run_ids[-1]) + 1
        unique_weights = np.zeros((num_unique, 2), dtype=np.float32)

        # Aggregation loop (np.add.at is slow, simple loop might be faster for CPU or stick to np.add.at)
        np.add.at(unique_weights, run_ids, sorted_weights)

        change_points = np.flatnonzero(boundaries) + 1
        unique_indices = np.concatenate((np.array([0]), change_points))
        unique_patterns_arr = sorted_patterns[unique_indices]

        good_pat_count = 0
        bad_pat_count = 0

        for i in range(num_unique):
            w_good = unique_weights[i, 0]
            w_bad = unique_weights[i, 1]
            pat = unique_patterns_arr[i].tolist()
            try:
                if (good_wt * w_good) < thresh:
                    self.insert_pattern(pat, MAX_VAL, pat_dot)
                    bad_pat_count += 1
                elif (good_wt * w_good - bad_wt * w_bad) >= thresh:
                    self.insert_pattern(pat, hyph_level, pat_dot)
                    good_pat_count += 1
            except RuntimeError:
                pass

        return good_pat_count, bad_pat_count

    def export_patterns(self, filename):
        patterns_list = []
        self._collect_patterns(self.trie_root, [], patterns_list)
        print(f"💾 Saving {len(patterns_list)} patterns to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("% Generated by OrthosColab\n")
            f.write("\\patterns{\n")
            for pat_str, val, dot in sorted(patterns_list, key=lambda x: x[0]):
                pat_chars = list(pat_str)
                out = []
                for i, ch in enumerate(pat_chars):
                    if i == dot:
                        out.append(str(val))
                    out.append(ch)
                if dot == len(pat_chars):
                    out.append(str(val))
                f.write(''.join(out) + '\n')
            f.write("}\n")

    def _collect_patterns(self, s, prefix, patterns_list):
        for c in range(self.cmin, self.cmax + 1):
            t = s + c
            if t < len(self.trie_c) and self.trie_c[t] == c:
                new_prefix = prefix + [c]
                h = int(self.trie_r[t])
                while h > 0:
                    val = self.ops_val[h]
                    dot = self.ops_dot[h]
                    if val > 0 and val < MAX_VAL:
                        pat_str = ''.join(self.xext[ch] if ch < len(self.xext) else '?' for ch in new_prefix)
                        patterns_list.append((pat_str, val, dot))
                    h = self.ops_op[h]
                child = int(self.trie_l[t])
                if child > 0:
                    self._collect_patterns(child, new_prefix, patterns_list)

    def validate_logic(self):
        """
        Runs a small deterministic test to validate logic without GPU (if possible)
        or with GPU if available.
        """
        print("\n🧪 Running Self-Test Validation...")

        # 1. Initialize Dummy Dictionary
        # We need a small dictionary to test hyphenation
        # Let's say we have the word 'foobar' and we want to break it as foo-bar.
        # So words: .foobar.
        # dots:     00000000  (Assuming no hyphen in input dictionary)
        # dotw:     11111111

        # Reset engine
        self.words = None
        self.word_lens = None
        self.dots = None
        self.dotw = None

        # Ensure 'a' exists (and others)
        for char in "abcdefghijklmnopqrstuvwxyz.":
            code = ord(char)
            if code not in self.xint:
                 self.xext.append(char)
                 self.xint[code] = len(self.xext) - 1
                 self.xclass[code] = LETTER_CLASS
                 self.cmax = len(self.xext) - 1

        # Reset Trie
        self.trie_c[:] = 0
        self.trie_l[:] = 0
        self.trie_r[:] = 0
        self.trie_taken[:] = 0
        self.ops_val[:] = 0
        self.ops_dot[:] = 0
        self.ops_op[:] = 0
        self.op_count = 0
        self.trie_max = 0
        self.trie_bmax = 0
        self.init_pattern_trie()

        # Insert pattern '1b' (hyphenate before b)
        # In 'foobar', b is at index 4 (0-based: . f o o b a r .)
        # . is 0, f is 1...
        # Let's use internal indices.

        # Construct a dummy dictionary entry manually WITH A HYPHEN
        # Word: "foobar" -> .foo-bar.
        # We represent this as ".foobar." with dots array having a hyphen mark.
        word_str = ".foobar."
        word_ints = [self.xint[ord(c)] for c in word_str]

        self.words = np.array([word_ints + [0]*(MAX_LEN-len(word_ints))], dtype=np.uint32, order='F')
        self.word_lens = np.array([len(word_ints)], dtype=np.int32)

        # Initialize dots: All NO_HYF except at 'o' (index 3) which means hyphen AFTER 'o' (before 'b')
        self.dots = np.zeros_like(self.words) + NO_HYF
        # 0 1 2 3 4 5 6 7
        # . f o o b a r .
        # Hyphen after 2nd 'o' (index 3).
        self.dots[0, 3] = IS_HYF

        self.dotw = np.ones_like(self.words)

        print("   Dictionary initialized with '.foo-bar.' (hyphen before b).")
        print("   Running generation (Level 1) to rediscover this hyphen...")

        # Run do_dictionary (Level 1)
        # It should find patterns that explain this hyphen.
        # We look for patterns of length 1, dot 0 (i.e. 'b' at the hyphen).
        good, bad = self.do_dictionary_gpu(
            pat_len=1, pat_dot=0, hyph_level=1,
            good_wt=1, bad_wt=1, thresh=1,
            left_hyphen_min=1, right_hyphen_min=1
        )

        if good > 0:
            print(f"   ✅ Logic Test Passed: Found {good} good patterns (Expected > 0).")
        else:
            print(f"   ❌ Logic Test Failed: Found {good} good patterns.")

        # Verify '1b' was inserted
        pats = []
        self._collect_patterns(self.trie_root, [], pats)
        # We expect ('b', 1, 0)
        found_b = False
        for p, v, d in pats:
            if p == 'b' and v == 1 and d == 0:
                found_b = True

        if not found_b:
            print(f"   ℹ️ Patterns found: {pats}")

        if found_b:
             print("   ✅ Pattern Verification Passed: '1b' (b, 1, 0) was generated.")
        else:
             print("   ❌ Pattern Verification Failed: '1b' not generated.")


# ==========================================
# MAIN RUNNER
# ==========================================
def input_int_pair(prompt, vmin, vmax):
    while True:
        try:
            line = input(prompt)
            parts = line.strip().split()
            if len(parts) >= 2:
                n1, n2 = int(parts[0]), int(parts[1])
            elif len(parts) == 1:
                n1 = int(parts[0])
                n2 = int(input("  ... second value: "))
            else:
                continue

            if n1 >= vmin and n2 <= vmax:
                return n1, n2
            else:
                print(f"Values must be between {vmin} and {vmax}")
        except ValueError:
            print("Invalid input.")

def input_gbt():
    while True:
        try:
            line = input("good weight, bad weight, threshold: ")
            parts = line.strip().split()
            if len(parts) == 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
            else:
                print("Please enter 3 integers.")
        except ValueError:
            pass

def main():
    parser = argparse.ArgumentParser(description="Orthos PatGen on GPU")
    parser.add_argument("dictionary", nargs='?', help="Path to dictionary file")
    parser.add_argument("pattern_in", nargs='?', help="Path to input pattern file")
    parser.add_argument("pattern_out", nargs='?', help="Path to output pattern file")
    parser.add_argument("--validate", action="store_true", help="Run validation logic only")
    args = parser.parse_args()

    # Mount Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass

    engine = OrthosEngine()

    if args.validate:
        engine.validate_logic()
        return

    if not args.dictionary or not args.pattern_in or not args.pattern_out:
        parser.print_help()
        sys.exit(1)

    # Security Check: Output Path
    try:
        engine.validate_output_file(args.pattern_out)
    except Exception as e:
        print(f"\n❌ Security Error: {e}")
        sys.exit(1)

    # Load Data
    engine.load_dictionary(args.dictionary)
    engine.load_patterns(args.pattern_in)

    print("\n=== Config ===")
    left_min, right_min = input_int_pair("left_hyphen_min, right_hyphen_min: ", 1, 15)

    hyph_start, hyph_finish = input_int_pair("hyph_start, hyph_finish: ", 1, MAX_VAL - 1)

    for level in range(hyph_start, hyph_finish + 1):
        print(f"\n--- Level {level} ---")
        pat_start, pat_finish = input_int_pair("pat_start, pat_finish: ", 1, MAX_DOT)

        good_wt, bad_wt, thresh = input_gbt()

        # Generation Loop
        more_this_level = [True] * (MAX_DOT + 1)

        # Generate Level
        print(f"🚀 Running generation for Level {level}...")
        for j in range(pat_start, pat_finish + 1):
            pat_len = j
            pat_dot = pat_len // 2
            dot1 = pat_dot * 2

            while True:
                pat_dot = dot1 - pat_dot
                dot1 = pat_len * 2 - dot1 - 1

                if more_this_level[pat_dot]:
                    print(f"  > Processing Length {pat_len}, Dot {pat_dot}")
                    good, bad = engine.do_dictionary_gpu(
                        pat_len, pat_dot, level,
                        good_wt, bad_wt, thresh,
                        left_min, right_min
                    )
                    print(f"    Result: +{good} good, -{bad} bad")

                if pat_dot == pat_len:
                    break

        engine.delete_bad_patterns()

    # Save
    engine.export_patterns(args.pattern_out)
    print("\n✅ Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {type(e).__name__} - {str(e)}")
        sys.exit(1)
