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
import numpy as np

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
            print("   Dependencies already installed.")
        except ImportError:
            print("   Installing CuPy and Numba...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
            print("   Installation complete.")
    else:
        # Local environment check
        try:
            import cupy
            import numba
        except ImportError:
            print("⚠️  Warning: CuPy or Numba not found. GPU acceleration will not work.")
            print("   Please install them: pip install numpy numba cupy-cuda12x")

install_dependencies()

# ==========================================
# IMPORTS
# ==========================================
try:
    import cupy as cp
    from numba import cuda, int32, uint32, float32, uint8
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

    @cuda.jit
    def kernel_update_dots(dots, hvals, dotw, word_lens, hyf_min, hyf_max,
                           found_hyf, err_hyf, is_hyf,
                           good_count, bad_count, miss_count):
        """
        Equivalent to change_dots() in orthos.js.
        Updates the 'dots' array based on calculated 'hvals'.
        Also aggregates counts (requires atomic add if we want global stats,
        but for pattern generation we mainly need the updated 'dots' state).
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

            # Accumulate stats (Optional: for reporting only)
            # We use atomic adds to global counters if needed,
            # but for logic correctness, modifying 'dots' is the key.
            # d_val = dots[idx, dpos]
            # weight = dotw[idx, dpos]
            # if d_val == found_hyf:
            #     cuda.atomic.add(good_count, 0, weight)
            # elif d_val == err_hyf:
            #     cuda.atomic.add(bad_count, 0, weight)
            # elif d_val == is_hyf:
            #     cuda.atomic.add(miss_count, 0, weight)

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
                    out_idx = idx * MAX_LEN + dpos

                    # Check bounds
                    if spos + 1 >= 0 and spos + 1 + pat_len <= wlen:
                        for k in range(pat_len):
                            out_patterns[out_idx, k] = words[idx, spos + 1 + k]

                        weight = dotw[idx, dpos]

                        if is_good:
                            out_weights[out_idx, 0] = weight
                            out_weights[out_idx, 1] = 0.0
                        else:
                            out_weights[out_idx, 0] = 0.0
                            out_weights[out_idx, 1] = float(weight)

                        out_mask[out_idx] = 1


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
        
        self.trie_c = np.zeros(self.trie_size, dtype=np.int64)
        self.trie_l = np.zeros(self.trie_size, dtype=np.int64)
        self.trie_r = np.zeros(self.trie_size, dtype=np.int64)
        self.trie_taken = np.zeros(self.trie_size, dtype=np.uint8)
        self.ops = [{'val':0, 'dot':0, 'op':0} for _ in range(self.max_ops + 1)] # 1-based

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

    def sync_trie_to_gpu(self):
        if not self.gpu_available: return

        limit = self.trie_max + 1

        self.d_trie_c = cp.asarray(self.trie_c[:limit].astype(np.int32))
        self.d_trie_l = cp.asarray(self.trie_l[:limit].astype(np.int32))
        self.d_trie_r = cp.asarray(self.trie_r[:limit].astype(np.int32))

        ops_val = np.zeros(MAX_OPS + 1, dtype=np.uint8)
        ops_dot = np.zeros(MAX_OPS + 1, dtype=np.uint8)
        ops_op = np.zeros(MAX_OPS + 1, dtype=np.uint32)

        for i in range(1, MAX_OPS + 1):
            ops_val[i] = self.ops[i]['val']
            ops_dot[i] = self.ops[i]['dot']
            ops_op[i] = self.ops[i]['op']

        self.d_ops_val = cp.asarray(ops_val)
        self.d_ops_dot = cp.asarray(ops_dot)
        self.d_ops_op = cp.asarray(ops_op)

    def sync_dictionary_to_gpu(self):
        if not self.gpu_available: return

        if (self.d_words is not None): return

        print("   [GPU] Uploading dictionary to VRAM...")
        self.d_words = cp.asarray(self.words, order='F')
        self.d_word_lens = cp.asarray(self.word_lens)
        self.d_dots = cp.asarray(self.dots, order='F')
        self.d_dotw = cp.asarray(self.dotw, order='F')

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
        h = ((next_op + 313 * dot + 361 * val) % MAX_OPS) + 1
        while True:
            if self.ops[h]['val'] == 0:
                self.op_count += 1
                if self.op_count == MAX_OPS:
                    raise RuntimeError("Outputs Overflow")
                self.ops[h]['val'] = val
                self.ops[h]['dot'] = dot
                self.ops[h]['op'] = next_op
                return h
            if (self.ops[h]['val'] == val and self.ops[h]['dot'] == dot and self.ops[h]['op'] == next_op):
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

    def load_dictionary(self, filepath):
        print(f"📖 Loading dictionary from {filepath}...")
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

        self.words = np.array(words_list, dtype=np.uint32, order='F')
        self.dots = np.array(dots_list, dtype=np.uint32, order='F')
        self.dotw = np.array(dotw_list, dtype=np.uint32, order='F')
        self.word_lens = np.array(lens_list, dtype=np.int32)

        # Reset Cache
        self.d_words = None

        # Reset Trie (new chars might have been added)
        self.trie_c = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_l = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_r = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_taken = np.zeros(TRIE_SIZE, dtype=np.uint8)
        self.ops = [{'val':0, 'dot':0, 'op':0} for _ in range(MAX_OPS + 1)]
        self.op_count = 0
        self.trie_max = 0
        self.trie_bmax = 0
        self.init_pattern_trie()

        print(f"   Loaded {len(words_list)} words.")

    def load_patterns(self, filepath):
        print(f"📖 Loading patterns from {filepath}...")
        if not os.path.exists(filepath):
            print(f"   File not found. Starting with empty patterns.")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            # Simple line based reading for now or TeX format?
            # Orthos.js read_patterns handled simple line based where chars have classes.
            # Assuming TeX-like or simple list.
            # Let's support standard TeX format as seen in orthos_gpu.py
            content = f.read()

        # Parse TeX patterns
        import re
        # This is a basic parser
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
                    ops_list.append((self.ops[h]['val'], self.ops[h]['dot'], self.ops[h]['op']))
                    h = self.ops[h]['op']
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
        return s

    def delete_bad_patterns(self):
        self.delete_patterns(self.trie_root)
        for h in range(1, MAX_OPS + 1):
            if self.ops[h]['val'] == MAX_VAL:
                self.ops[h]['val'] = 0
        self.qmax_thresh = 7

    def do_dictionary_gpu(self, pat_len, pat_dot, hyph_level, good_wt, bad_wt, thresh, left_hyphen_min, right_hyphen_min):
        if not self.gpu_available:
            print("❌ GPU not available. Skipping generation.")
            return 0, 0

        self.sync_trie_to_gpu()
        self.sync_dictionary_to_gpu()

        n_words = self.words.shape[0]
        d_hvals = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')
        d_no_more = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')

        # We need a copy of dots because update_dots modifies it in-place for the kernel run
        # but we don't want to persist these changes if the algorithm requires fresh start (actually orthos.js updates in memory?)
        # orthos.js: "change_dots() -> dots[dpos] += 1". This persists for the word.
        # But orthos.js reads dictionary from file *every* do_dictionary() pass, resetting the state!
        # Thus, we MUST start with the original 'dots' every time.
        d_dots_working = cp.array(self.d_dots, copy=True) # Make a working copy from the cached read-only dots

        threadsperblock = 128
        blockspergrid = (n_words + (threadsperblock - 1)) // threadsperblock

        # 1. Hyphenate (fill hvals based on CURRENT trie)
        kernel_hyphenate[blockspergrid, threadsperblock](
            self.d_words, self.d_word_lens,
            self.d_trie_c, self.d_trie_l, self.d_trie_r,
            self.d_ops_val, self.d_ops_dot, self.d_ops_op,
            self.trie_root,
            d_hvals, d_no_more,
            pat_len, pat_dot, hyph_level, MAX_VAL
        )

        hyf_min = left_hyphen_min + 1
        hyf_max = right_hyphen_min + 1

        # 2. Update Dots (Simulate change_dots: update dots array based on hvals)
        # Dummy counters for now, we don't display stats during generation usually to save time
        d_good_count = cp.array([0], dtype=cp.uint32)
        d_bad_count = cp.array([0], dtype=cp.uint32)
        d_miss_count = cp.array([0], dtype=cp.uint32)

        kernel_update_dots[blockspergrid, threadsperblock](
            d_dots_working, d_hvals, self.d_dotw, self.d_word_lens,
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

        d_out_patterns = cp.zeros((n_words * MAX_LEN, pat_len), dtype=cp.uint32)
        d_out_weights = cp.zeros((n_words * MAX_LEN, 2), dtype=cp.float32)
        d_out_mask = cp.zeros((n_words * MAX_LEN), dtype=cp.uint8)

        # 3. Extract Candidates using the UPDATED dots
        kernel_extract_candidates[blockspergrid, threadsperblock](
            self.d_words, self.d_word_lens, d_dots_working, self.d_dotw, d_no_more,
            pat_len, pat_dot, good_dot, bad_dot,
            dot_min, dot_max,
            d_out_patterns, d_out_weights, d_out_mask
        )

        # Aggregation
        mask = d_out_mask.astype(bool)
        valid_patterns = d_out_patterns[mask]
        valid_weights = d_out_weights[mask]

        if valid_patterns.shape[0] == 0:
            return 0, 0

        # Lexsort for reduction
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
                    val = self.ops[h]['val']
                    dot = self.ops[h]['dot']
                    if val > 0 and val < MAX_VAL:
                        pat_str = ''.join(self.xext[ch] if ch < len(self.xext) else '?' for ch in new_prefix)
                        patterns_list.append((pat_str, val, dot))
                    h = self.ops[h]['op']
                child = int(self.trie_l[t])
                if child > 0:
                    self._collect_patterns(child, new_prefix, patterns_list)

    def validate_logic(self):
        """
        Runs a small deterministic test to validate logic without GPU (if possible)
        or with GPU if available.
        This mirrors a small subset of PatGen to ensure '1a' pattern works.
        """
        print("\n🧪 Running Self-Test Validation...")

        # 1. Ensure 'a' exists in char map
        if ord('a') not in self.xint:
             self.xext.append('a')
             self.xint[ord('a')] = len(self.xext) - 1
             self.xclass[ord('a')] = LETTER_CLASS
             self.cmax = len(self.xext) - 1

        # 2. Reset Trie with updated char map
        self.trie_c[:] = 0
        self.trie_l[:] = 0
        self.trie_r[:] = 0
        self.trie_taken[:] = 0
        self.ops = [{'val':0, 'dot':0, 'op':0} for _ in range(self.max_ops + 1)]
        self.op_count = 0
        self.trie_max = 0
        self.trie_bmax = 0
        self.init_pattern_trie()

        # 3. Add '1a' pattern manually
        idx_a = self.xint[ord('a')]
        self.insert_pattern([idx_a], 1, 0) # 1a

        # 3. Test Hyphenation on 'a' (should be '-a')
        # We can't easily run the GPU kernel here without context,
        # but we can check if the Trie contains the pattern.
        pats = []
        self._collect_patterns(self.trie_root, [], pats)

        found = False
        for p, v, d in pats:
            if p == 'a' and v == 1 and d == 0:
                found = True

        if found:
            print("   ✅ Pattern Insertion Test Passed (Found '1a')")
        else:
            print("   ❌ Pattern Insertion Test Failed")

        print("   (Full functional test requires GPU execution flow)")


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
    parser.add_argument("dictionary", help="Path to dictionary file")
    parser.add_argument("pattern_in", help="Path to input pattern file")
    parser.add_argument("pattern_out", help="Path to output pattern file")
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
    main()
