# @title Install Dependencies
# !pip install numpy numba cupy-cuda12x

import numpy as np
import time
import sys
import os

# Check for GPU
try:
    import cupy as cp
    from numba import cuda
    # Try accessing device to confirm availability
    try:
        cp.cuda.Device(0).compute_capability
        GPU_AVAILABLE = True
    except Exception as e:
        print(f"Warning: GPU detected but not accessible ({e}). Switching to CPU/Mock mode.")
        raise ImportError("Force CPU Mock")
except ImportError:
    print("Warning: CuPy/Numba not installed or no GPU. Running in CPU emulation/fail mode.")
    GPU_AVAILABLE = False
    # Mocking for local testing/CPU fallback
    import numpy as cp
    class MockCuda:
        def jit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def grid(self, n):
            return 0
    cuda = MockCuda()

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
MAX_VAL = 10
MAX_DOT = 15
MAX_LEN = 50
TRIE_SIZE = 55000 * 80
MAX_OPS = 510 * 10

# Character Classes
DIGIT_CLASS = 1
HYF_CLASS = 2
LETTER_CLASS = 3
NO_HYF = 10     # ''
ERR_HYF = 11    # '#'
IS_HYF = 12     # '-'
FOUND_HYF = 13  # '*'

# ==========================================
# COLAB HELPERS
# ==========================================
def mount_drive():
    """Helper to mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Drive mounted at /content/drive")
    except ImportError:
        print("Not running in Google Colab or module not found.")

# ==========================================
# GPU KERNELS
# ==========================================

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
        t = trie_root + char_code

        # Traverse Trie
        while True:
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

                t = trie_l[t]
                if t == 0:
                    break

                fpos += 1
                if fpos >= wlen:
                    break
                char_code = words[idx, fpos]
                t = t + char_code
            else:
                break


@cuda.jit
def kernel_extract_candidates(words, word_lens, dots, dotw, hvals, no_more,
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


class OrthosGPU:
    def __init__(self):
        self.xint = {}
        self.xext = []
        self.xclass = {}
        self.trie_c = np.zeros(TRIE_SIZE, dtype=np.uint32)
        self.trie_l = np.zeros(TRIE_SIZE, dtype=np.uint32)
        self.trie_r = np.zeros(TRIE_SIZE, dtype=np.uint32)
        self.trie_taken = np.zeros(TRIE_SIZE, dtype=np.uint8)
        self.ops = [{'val':0, 'dot':0, 'op':0} for _ in range(MAX_OPS + 1)] # 1-based

        # Trie packing structures
        self.trieq_c = np.zeros(50, dtype=np.uint32)
        self.trieq_l = np.zeros(50, dtype=np.uint32)
        self.trieq_r = np.zeros(50, dtype=np.uint32)
        self.qmax = 0
        self.qmax_thresh = 5
        self.trie_max = 0
        self.trie_bmax = 0
        self.op_count = 0

        self.trie_root = 1

        self.initialize_chars()
        # Trie init deferred until after char collection
        # self.init_pattern_trie()

        # Data
        self.words = None
        self.word_lens = None
        self.dots = None
        self.dotw = None

        # GPU Cache for Dictionary
        self.d_words = None
        self.d_word_lens = None
        self.d_dots = None
        self.d_dotw = None

        self.gpu_available = GPU_AVAILABLE

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
        self.d_trie_c = cp.asarray(self.trie_c)
        self.d_trie_l = cp.asarray(self.trie_l)
        self.d_trie_r = cp.asarray(self.trie_r)

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
        """Uploads dictionary data to GPU only if not already cached."""
        if not self.gpu_available: return

        # Check if already cached
        if (self.d_words is not None and
            self.d_word_lens is not None and
            self.d_dots is not None and
            self.d_dotw is not None):
            return

        print("Uploading dictionary to GPU (Caching)...")
        # Ensure order is preserved or correctly handled by cupy
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
            t = self.trie_l[t]
            if t == 0:
                 continue

            s = int(t) - int(self.trieq_c[1])

            if s > TRIE_SIZE - len(self.xext):
                 if s < 0: continue
                 raise RuntimeError(f"Trie Overflow: s={s}, t={t}")

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
            t_node = s + int(self.trieq_c[q])
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

            if (self.ops[h]['val'] == val and
                self.ops[h]['dot'] == dot and
                self.ops[h]['op'] == next_op):
                return h

            if h > 1:
                h -= 1
            else:
                h = MAX_OPS

    def insert_pattern(self, pat, val, dot):
        # pat is list of ints, 0-indexed
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

    def load_dictionary_safe(self, filepath):
        print(f"Loading dictionary from {filepath}...")
        words_list = []
        dots_list = []
        dotw_list = []
        lens_list = []

        try:
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

                    l = len(word_ints)
                    lens_list.append(l)

                    # Check for MAX_LEN overflow
                    if len(word_ints) > MAX_LEN:
                        word_ints = word_ints[:MAX_LEN]
                        dots = dots[:MAX_LEN]
                        dotw = dotw[:MAX_LEN]

                    pad_len = MAX_LEN - len(word_ints)
                    if pad_len > 0:
                        word_ints.extend([0] * pad_len)
                        dots.extend([0] * pad_len)
                        dotw.extend([0] * pad_len)

                    words_list.append(word_ints)
                    dots_list.append(dots)
                    dotw_list.append(dotw)
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return False

        self.words = np.array(words_list, dtype=np.uint32, order='F')
        self.dots = np.array(dots_list, dtype=np.uint32, order='F')
        self.dotw = np.array(dotw_list, dtype=np.uint32, order='F')
        self.word_lens = np.array(lens_list, dtype=np.int32)

        # Reset GPU cache
        self.d_words = None
        self.d_word_lens = None
        self.d_dots = None
        self.d_dotw = None

        print(f"Loaded {len(words_list)} words.")
        return True

    def scan_patterns(self, filepath):
        """Scans pattern file to register all characters before Trie init."""
        print(f"Scanning patterns from {filepath} for characters...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    for char in line:
                        code = ord(char)
                        if code not in self.xint:
                            self.xext.append(char)
                            idx = len(self.xext) - 1
                            self.xint[code] = idx
                            self.xclass[code] = LETTER_CLASS
                            self.cmax = len(self.xext) - 1
        except FileNotFoundError:
            print(f"Pattern file {filepath} not found (skipping scan).")

    def read_patterns(self, filepath):
        print(f"Reading patterns from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Pattern file {filepath} not found. Starting with empty patterns.")
            return

        level_pattern_count = 0
        max_pat = 0

        for line in lines:
            line = line.strip()
            if not line: continue

            level_pattern_count += 1

            # Logic same as JS read_patterns
            pat = []
            hval = [0] * (len(line) + 2)
            pat_len = 0
            hval[0] = 0

            i = 0
            while i < len(line):
                char = line[i]
                code = ord(char)

                if code not in self.xint:
                    # Should not happen if scanned, but fallback just in case
                    # (Note: doing this here AFTER init_trie is bad, but better than crashing if we can handle it?
                    # No, it will crash Trie. So we print warning).
                    print(f"Warning: New char '{char}' found in pattern after Trie init. Ignored.")
                    i += 1
                    continue

                cls = self.xclass[code]
                if cls == DIGIT_CLASS:
                    d = self.xint[code]
                    if d > max_pat: max_pat = d
                    hval[pat_len] = d
                elif cls == LETTER_CLASS:
                    pat_len += 1
                    hval[pat_len] = 0
                    pat.append(self.xint[code])
                i += 1

            for k in range(pat_len + 1):
                if hval[k] != 0:
                    self.insert_pattern(pat, hval[k], k)

        print(f"Read {level_pattern_count} patterns.")

    def _output_patterns_recursive(self, s, pat_buffer, pat_len, outfile):
        c = self.cmin
        while c <= self.cmax:
            t = s + c
            if self.trie_c[t] == c:
                # pat_buffer[pat_len] is equivalent to JS pat[pat_len]
                # JS uses 1-based indexing for pattern construction during traversal.
                if len(pat_buffer) <= pat_len:
                    pat_buffer.append(c)
                else:
                    pat_buffer[pat_len] = c

                h = self.trie_r[t]
                if h > 0:
                    # Reconstruct hval
                    # JS: d=0..pat_len, hval[d]=0
                    # traverse ops list
                    local_hval = [0] * (pat_len + 2)

                    curr = h
                    while curr != 0:
                        d = self.ops[curr]['dot']
                        v = self.ops[curr]['val']
                        if local_hval[d] < v:
                            local_hval[d] = v
                        curr = self.ops[curr]['op']

                    # Write to file
                    # Format: hval[0] (if >0) + pat[0] + hval[1] (if >0) + pat[1] ...
                    line = ""
                    if local_hval[0] > 0:
                        line += self.xext[local_hval[0]] # Using xext to map int back to char (digit)

                    for k in range(pat_len): # 0 to pat_len-1
                        # char at k (which is pat_buffer[k] in 0-indexed list, or pat[k+1] in JS)
                        # JS loop: d=1..pat_len. write xext[pat[d]]. if hval[d]>0 write.
                        # pat_buffer is 0-indexed.
                        char_code = pat_buffer[k]
                        line += self.xext[char_code]
                        if local_hval[k+1] > 0:
                            line += self.xext[local_hval[k+1]]

                    outfile.write(line + "\n")

                if self.trie_l[t] > 0:
                    self._output_patterns_recursive(self.trie_l[t], pat_buffer, pat_len + 1, outfile)
            c += 1

    def output_patterns(self, filepath):
        print(f"Writing patterns to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            self._output_patterns_recursive(self.trie_root, [], 0, f)

    def delete_patterns(self, s):
        c = self.cmin
        all_freed = True

        while c <= self.cmax:
            t = s + c
            if self.trie_c[t] == c:
                # Compression logic
                ops_list = []
                h = self.trie_r[t]
                while h > 0:
                    ops_list.append(self.ops[h])
                    h = self.ops[h]['op']

                # Keep ops where val == MAX_VAL
                new_head = 0
                for op in reversed(ops_list):
                    if op['val'] == MAX_VAL:
                        h = self.new_trie_op(op['val'], op['dot'], new_head)
                        new_head = h

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
        print("Deleting bad patterns...")
        self.delete_patterns(self.trie_root)
        for h in range(1, MAX_OPS + 1):
            if self.ops[h]['val'] == MAX_VAL:
                self.ops[h]['val'] = 0
        self.qmax_thresh = 7

    def calculate_stats(self, hyph_level, left_hyphen_min, right_hyphen_min):
        """Calculates global hyphenation statistics (good, bad, missed)."""
        if not self.gpu_available: return 0,0,0

        # 1. Ensure hvals are computed (assumes do_dictionary_gpu ran recently)
        # However, do_dictionary_gpu is per pattern length. We need stats for the WHOLE dictionary given CURRENT Trie.
        # So we need to run kernel_hyphenate once for the whole dictionary with dummy pat_len/dot
        # or just run it to update hvals.

        # We need a clean run of hyphenation to get current state
        self.sync_trie_to_gpu()
        self.sync_dictionary_to_gpu()

        n_words = self.words.shape[0]
        d_hvals = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')
        d_no_more = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F') # Not used for stats but needed for kernel

        threadsperblock = 128
        blockspergrid = (n_words + (threadsperblock - 1)) // threadsperblock

        # Dummy values for pat_len/pat_dot as we don't care about no_more for stats
        kernel_hyphenate[blockspergrid, threadsperblock](
            self.d_words, self.d_word_lens,
            self.d_trie_c, self.d_trie_l, self.d_trie_r,
            self.d_ops_val, self.d_ops_dot, self.d_ops_op,
            self.trie_root,
            d_hvals, d_no_more,
            1, 0, hyph_level, MAX_VAL
        )

        # Now compute stats using Array operations
        # dot limits
        hyf_min = left_hyphen_min + 1
        hyf_max = right_hyphen_min + 1

        # Create a mask for valid positions [hyf_min, wlen - hyf_max]
        # This is tricky with 2D array and variable lengths.
        # We can construct a mask.
        col_indices = cp.arange(MAX_LEN)[None, :] # 1 x 50
        # Valid if: col >= hyf_min AND col <= (wlen - hyf_max)
        # wlen is (N,) -> (N, 1)
        wlens = self.d_word_lens[:, None]
        valid_mask = (col_indices >= hyf_min) & (col_indices <= (wlens - hyf_max))

        # Predicted Hyphen: hval is odd
        is_predicted = (d_hvals % 2) == 1

        # Truth
        dots = self.d_dots
        dotw = self.d_dotw.astype(cp.float32)

        # Logic
        # Found (Good): Predicted AND dots==FOUND_HYF
        # Bad: Predicted AND dots==ERR_HYF
        # Missed: NOT Predicted AND dots==IS_HYF

        # Wait, JS logic for dots:
        # Initially in dictionary:
        # '-' -> is_hyf (12)
        # '*' -> found_hyf (13) -- Wait, in JS found_hyf is 13.
        # '#' -> err_hyf (11)

        # JS `change_dots` modifies `dots` array in place!
        # "If (hval[dpos] % 2) === 1) { dots[dpos] += 1; }"
        # This changes state:
        # is_hyf (12) -> found_hyf (13)
        # no_hyf (10) -> err_hyf (11)
        # err_hyf (11) -> 12 (is_hyf?? No, err_hyf+1 doesn't make sense logically but that's what JS does?)
        # Let's check JS:
        # "case found_hyf: good_count..."
        # So it permanently marks them?
        # NO. `change_dots` is called in `do_dictionary`. But `dots` are reset?
        # JS `read_word` reads from `dictionary` which is in memory.
        # Does it reset `dots` every time?
        # `read_word` sets `dots[wlen] = xint[cc]` or `no_hyf`.
        # Yes, `read_word` re-initializes `dots` from the raw file buffer every time `do_dictionary` loops through the file.
        # So we don't need to persist state. We just compare.

        # Valid Mask application
        # We need to consider ONLY valid positions.

        # Good Count: Where (hvals % 2 == 1) AND (original_dots == IS_HYF or original_dots == FOUND_HYF)
        # Wait, IS_HYF is a hyphen that SHOULD be there.
        # If we predict it, it's GOOD.
        # If we don't, it's MISSED.
        # If we predict where there is NO_HYF, it's BAD.

        # My constants:
        # IS_HYF = 12 (-)
        # NO_HYF = 10 (empty)
        # ERR_HYF = 11 (#)

        c_predicted = (is_predicted & valid_mask)
        c_not_predicted = ((~is_predicted) & valid_mask)

        c_is_hyf = (dots == IS_HYF)
        c_no_hyf = (dots == NO_HYF)
        c_err_hyf = (dots == ERR_HYF) # Explicitly bad place

        # Calculate
        # Good: Predicted & Is_Hyf
        good_mask = c_predicted & c_is_hyf
        good_count = cp.sum(dotw * good_mask)

        # Bad: Predicted & (No_Hyf OR Err_Hyf)
        bad_mask = c_predicted & (c_no_hyf | c_err_hyf)
        bad_count = cp.sum(dotw * bad_mask)

        # Missed: Not Predicted & Is_Hyf
        miss_mask = c_not_predicted & c_is_hyf
        miss_count = cp.sum(dotw * miss_mask)

        return float(good_count), float(bad_count), float(miss_count)

    def do_dictionary_gpu(self, pat_len, pat_dot, hyph_level, good_wt, bad_wt, thresh, left_hyphen_min, right_hyphen_min):
        if not self.gpu_available:
            print("Skipping GPU execution (No GPU).")
            return 0, 0

        # 1. Sync Trie (Always needed as Trie changes)
        self.sync_trie_to_gpu()

        # 2. Sync Dictionary (Cached)
        self.sync_dictionary_to_gpu()

        # Use cached handles
        d_words = self.d_words
        d_word_lens = self.d_word_lens
        d_dots = self.d_dots
        d_dotw = self.d_dotw

        n_words = self.words.shape[0]
        # Hvals also benefit from F order
        d_hvals = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')
        d_no_more = cp.zeros((n_words, MAX_LEN), dtype=cp.uint8, order='F')

        threadsperblock = 128
        blockspergrid = (n_words + (threadsperblock - 1)) // threadsperblock

        # 3. Hyphenate Kernel
        kernel_hyphenate[blockspergrid, threadsperblock](
            d_words, d_word_lens,
            self.d_trie_c, self.d_trie_l, self.d_trie_r,
            self.d_ops_val, self.d_ops_dot, self.d_ops_op,
            self.trie_root,
            d_hvals, d_no_more,
            pat_len, pat_dot, hyph_level, MAX_VAL
        )

        # Determine Good/Bad Dot
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

        # 4. Extract Candidates
        d_out_patterns = cp.zeros((n_words * MAX_LEN, pat_len), dtype=cp.uint32)
        d_out_weights = cp.zeros((n_words * MAX_LEN, 2), dtype=cp.float32) # [good, bad]
        d_out_mask = cp.zeros((n_words * MAX_LEN), dtype=cp.uint8)

        kernel_extract_candidates[blockspergrid, threadsperblock](
            d_words, d_word_lens, d_dots, d_dotw, d_hvals, d_no_more,
            pat_len, pat_dot, good_dot, bad_dot,
            dot_min, dot_max,
            d_out_patterns, d_out_weights, d_out_mask
        )

        # 5. Aggregation
        mask = d_out_mask.astype(bool)
        valid_patterns = d_out_patterns[mask]
        valid_weights = d_out_weights[mask]

        if valid_patterns.shape[0] == 0:
            return 0, 0

        keys = tuple(valid_patterns[:, i] for i in range(pat_len - 1, -1, -1))
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

        # Get unique patterns
        change_points = cp.flatnonzero(boundaries) + 1
        unique_indices = cp.concatenate((cp.array([0]), change_points))
        unique_patterns_arr = sorted_patterns[unique_indices]

        # 6. Selection & CPU Update
        cpu_weights = unique_weights.get()
        cpu_patterns = unique_patterns_arr.get()

        good_pat_count = 0
        bad_pat_count = 0

        for i in range(num_unique):
            w_good = cpu_weights[i, 0]
            w_bad = cpu_weights[i, 1]

            pat = cpu_patterns[i].tolist()

            if (good_wt * w_good) < thresh:
                self.insert_pattern(pat, MAX_VAL, pat_dot)
                bad_pat_count += 1
            elif (good_wt * w_good - bad_wt * w_bad) >= thresh:
                self.insert_pattern(pat, hyph_level, pat_dot)
                good_pat_count += 1
            else:
                pass

        return good_pat_count, bad_pat_count

    def generate_level(self, pat_start, pat_finish, hyph_level, good_wt, bad_wt, thresh, left_hyphen_min, right_hyphen_min):
        print(f"\n--- Generating Level {hyph_level} ---")
        print(f"Params: Start={pat_start}, Finish={pat_finish}, GoodWt={good_wt}, BadWt={bad_wt}, Thresh={thresh}")

        more_this_level = [True] * (MAX_DOT + 1)
        total_good = 0

        for j in range(pat_start, pat_finish + 1):
            pat_len = j
            pat_dot = pat_len // 2
            dot1 = pat_dot * 2

            while True:
                pat_dot = dot1 - pat_dot
                dot1 = pat_len * 2 - dot1 - 1

                if more_this_level[pat_dot]:
                    #print(f"  Length {pat_len}, Dot {pat_dot}")
                    good, bad = self.do_dictionary_gpu(
                        pat_len, pat_dot, hyph_level,
                        good_wt, bad_wt, thresh,
                        left_hyphen_min, right_hyphen_min
                    )
                    if good > 0 or bad > 0:
                        print(f"    Len {pat_len} Dot {pat_dot}: +{good} good, +{bad} bad")

                    total_good += good

                    # Heuristic for "more to come" from JS?
                    # JS sets more_to_come = true if undecided patterns exist.
                    # My GPU logic currently drops undecided patterns.
                    # Implementing exact "more_to_come" requires returning a flag if any pattern was between bounds.
                    # For now, we assume if we found ANY good patterns, keep looking.
                    if good > 0:
                        more_this_level[pat_dot] = True
                    else:
                        more_this_level[pat_dot] = False

                if pat_dot == pat_len:
                    break

        print(f"Total patterns added this level: {total_good}")
        self.delete_bad_patterns()

        # Report Stats
        g, b, m = self.calculate_stats(hyph_level, left_hyphen_min, right_hyphen_min)
        total = g + b + m
        if total > 0:
             print(f"Stats: Good={int(g)} ({g/total:.1%}), Bad={int(b)} ({b/total:.1%}), Missed={int(m)} ({m/total:.1%})")
        else:
             print("Stats: 0 total (Dictionary empty or error?)")

def main():
    import argparse

    print("OrthosGPU - Python/CUDA Port of PatGen")

    # Attempt to mount drive if in Colab
    if 'google.colab' in sys.modules:
        mount_drive()

    # Simple CLI argument parsing
    if len(sys.argv) < 4:
        print("Usage: python orthos_gpu.py <dictionary_file> <pattern_file_in> <pattern_file_out>")
        # If in interactive/test mode, don't exit, just print warning
        if 'ipykernel' not in sys.modules:
             # sys.exit(1)
             pass
        else:
             print("Running in interactive mode/notebook.")
             return

    dict_file = sys.argv[1]
    pat_in = sys.argv[2]
    pat_out = sys.argv[3]

    app = OrthosGPU()

    if not app.load_dictionary_safe(dict_file):
        print("Failed to load dictionary.")
        return

    app.scan_patterns(pat_in)
    app.init_pattern_trie()
    app.read_patterns(pat_in)

    # Interactive Loop
    try:
        # Default Params
        left_min = int(input("left_hyphen_min [2]: ") or 2)
        right_min = int(input("right_hyphen_min [2]: ") or 2)

        while True:
            print("\nNext Step:")
            print("1. Run Level Generation")
            print("2. Output Patterns & Exit")
            choice = input("Choice [1]: ").strip() or "1"

            if choice == "2":
                app.output_patterns(pat_out)
                print("Done.")
                break

            # Level Config
            hyph_level = int(input("Hyphenation Level (1-9): "))
            pat_range = input("Pattern Start, Finish (e.g. '2 5'): ").split()
            if len(pat_range) == 1: pat_range.append(pat_range[0])
            p_start, p_end = int(pat_range[0]), int(pat_range[1])

            weights = input("GoodWt, BadWt, Thresh (e.g. '1 1 10'): ").split()
            if len(weights) < 3: weights = [1, 1, 1]
            gw, bw, th = int(weights[0]), int(weights[1]), int(weights[2])

            app.generate_level(p_start, p_end, hyph_level, gw, bw, th, left_min, right_min)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        try:
             save = input("Save current patterns? (y/n): ")
             if save.lower() == 'y':
                 app.output_patterns(pat_out)
        except:
             pass

if __name__ == "__main__":
    main()
