# @title Install Dependencies
# !pip install numpy numba cupy-cuda12x

import numpy as np
import cupy as cp
from numba import cuda, uint8, uint32, int32, float32
import time
import sys
import os

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
MAX_VAL = 10
MAX_DOT = 15
MAX_LEN = 50
TRIE_SIZE = 55000 * 200  # Increased for larger dictionaries
MAX_OPS = 510 * 50       # Increased for more patterns

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
        t = int32(trie_root) + int32(char_code)

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
        self.trie_c = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_l = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_r = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_taken = np.zeros(TRIE_SIZE, dtype=np.uint8)
        self.ops = [{'val':0, 'dot':0, 'op':0} for _ in range(MAX_OPS + 1)] # 1-based

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

        # GPU Cache for Dictionary
        self.d_words = None
        self.d_word_lens = None
        self.d_dots = None
        self.d_dotw = None

        self.gpu_available = False
        try:
            cuda.get_current_device()
            self.gpu_available = True
        except:
            print("Warning: No GPU detected. GPU operations will fail.")

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
        # Cast to int32 for GPU kernel compatibility (kernel uses int32 indexing)
        self.d_trie_c = cp.asarray(self.trie_c.astype(np.int32))
        self.d_trie_l = cp.asarray(self.trie_l.astype(np.int32))
        self.d_trie_r = cp.asarray(self.trie_r.astype(np.int32))

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
            t = int(self.trie_l[t])
            s = int(t) - int(self.trieq_c[1])
            # Check for overflow (negative s or too large)
            if s < 0 or s > TRIE_SIZE - len(self.xext):
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
                    # Truncate
                    word_ints = word_ints[:MAX_LEN]
                    dots = dots[:MAX_LEN]
                    dotw = dotw[:MAX_LEN]
                    # Warn?

                pad_len = MAX_LEN - len(word_ints)
                if pad_len > 0:
                    word_ints.extend([0] * pad_len)
                    dots.extend([0] * pad_len)
                    dotw.extend([0] * pad_len)

                words_list.append(word_ints)
                dots_list.append(dots)
                dotw_list.append(dotw)

        # Optimize memory layout for GPU coalescing: Use Fortran order (Column-Major)
        # This means words[idx, :] is not contiguous, but words[:, col] is.
        # Wait, for coalescing: Thread T reads words[T, 0]. Thread T+1 reads words[T+1, 0].
        # These addresses should be adjacent.
        # In Row-Major (C): words[T, 0] is at Base + T*RowSize. words[T+1, 0] is at Base + (T+1)*RowSize. Stride = RowSize. BAD.
        # In Col-Major (F): words[T, 0] is at Base + T. words[T+1, 0] is at Base + T + 1. Stride = 1. GOOD.

        self.words = np.array(words_list, dtype=np.uint32, order='F')
        self.dots = np.array(dots_list, dtype=np.uint32, order='F')
        self.dotw = np.array(dotw_list, dtype=np.uint32, order='F')
        self.word_lens = np.array(lens_list, dtype=np.int32) # Vector

        # Reset GPU cache
        self.d_words = None
        self.d_word_lens = None
        self.d_dots = None
        self.d_dotw = None

        # Reinitialize trie with the expanded character set
        # Reset trie arrays since cmax may have changed
        self.trie_c = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_l = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_r = np.zeros(TRIE_SIZE, dtype=np.int64)
        self.trie_taken = np.zeros(TRIE_SIZE, dtype=np.uint8)
        self.ops = [{'val':0, 'dot':0, 'op':0} for _ in range(MAX_OPS + 1)]
        self.op_count = 0
        self.trie_max = 0
        self.trie_bmax = 0
        self.init_pattern_trie()

        print(f"Loaded {len(words_list)} words with {self.cmax + 1} unique characters.")

        # Immediately upload dictionary to GPU to free CPU memory
        if self.gpu_available:
            print("Uploading dictionary to GPU...")
            self.d_words = cp.asarray(self.words, order='F')
            self.d_word_lens = cp.asarray(self.word_lens)
            self.d_dots = cp.asarray(self.dots, order='F')
            self.d_dotw = cp.asarray(self.dotw, order='F')
            # Keep CPU copies for evaluate() but could delete if GPU-only evaluation is implemented
            print(f"  ✅ Dictionary on GPU. GPU Memory used: {cp.get_default_memory_pool().used_bytes() / 1e6:.1f} MB")

    def load_patterns(self, filepath):
        """
        Load existing patterns from a TeX hyphenation file.
        
        Format expected: Standard TeX \patterns{...} format
        e.g., "1ba", "a1a", ".meng1", "2kh"
        
        Numbers indicate hyphenation values at that position.
        '.' indicates word boundary.
        """
        print(f"Loading seed patterns from {filepath}...")
        
        pattern_count = 0
        in_patterns = False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments
                if line.startswith('%'):
                    continue
                
                # Detect \patterns{ block
                if '\\patterns{' in line:
                    in_patterns = True
                    # Get content after \patterns{
                    line = line.split('\\patterns{', 1)[1]
                
                if not in_patterns:
                    continue
                
                # Detect end of patterns block
                if '}' in line:
                    line = line.split('}')[0]
                    in_patterns = False
                
                # Parse patterns from line (space-separated)
                tokens = line.split()
                for token in tokens:
                    if not token:
                        continue
                    
                    # Parse pattern: extract letters and digit positions
                    # e.g., "1ba" -> pattern ['b','a'], value=1 at dot position 0
                    # e.g., "a1a" -> pattern ['a','a'], value=1 at dot position 1
                    # e.g., ".meng1" -> pattern ['.','m','e','n','g'], value=1 at position 5
                    
                    letters = []
                    values = {}  # position -> value
                    
                    i = 0
                    letter_pos = 0
                    while i < len(token):
                        char = token[i]
                        if char.isdigit():
                            values[letter_pos] = int(char)
                            i += 1
                        else:
                            letters.append(char)
                            letter_pos += 1
                            i += 1
                    
                    if not letters:
                        continue
                    
                    # Convert to internal format
                    pat_ints = []
                    for ch in letters:
                        code = ord(ch)
                        if code not in self.xint:
                            # Add new character to alphabet
                            self.xext.append(ch)
                            idx = len(self.xext) - 1
                            self.xint[code] = idx
                            self.xclass[code] = LETTER_CLASS
                            self.cmax = len(self.xext) - 1
                        pat_ints.append(self.xint[code])
                    
                    # Insert patterns for each value position
                    for dot_pos, val in values.items():
                        if val > 0 and val < MAX_VAL:
                            try:
                                self.insert_pattern(pat_ints, val, dot_pos)
                                pattern_count += 1
                            except RuntimeError as e:
                                print(f"  Warning: Failed to insert pattern '{token}': {e}")
        
        print(f"  ✅ Loaded {pattern_count} seed patterns")
        return pattern_count

    def delete_patterns(self, s):
        c = self.cmin
        all_freed = True

        while c <= self.cmax:
            t = s + c
            if self.trie_c[t] == c:
                # Collect ops as TUPLES (value copies), not dict references
                ops_list = []
                h = int(self.trie_r[t])
                while h > 0:
                    # Store copies of values, not dictionary references!
                    ops_list.append((self.ops[h]['val'], self.ops[h]['dot'], self.ops[h]['op']))
                    h = self.ops[h]['op']

                # Keep GOOD patterns (val != MAX_VAL), remove bad ones (val == MAX_VAL)
                new_head = 0
                for val, dot, _ in reversed(ops_list):
                    if val != MAX_VAL and val > 0:  # Keep good patterns
                        h = self.new_trie_op(val, dot, new_head)
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

        # CuPy's lexsort expects a 2D array, not a tuple
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
        print(f"Generating Level {hyph_level}...")

        more_this_level = [True] * (MAX_DOT + 1)

        for j in range(pat_start, pat_finish + 1):
            pat_len = j
            pat_dot = pat_len // 2
            dot1 = pat_dot * 2

            while True:
                pat_dot = dot1 - pat_dot
                dot1 = pat_len * 2 - dot1 - 1

                if more_this_level[pat_dot]:
                    print(f"  Length {pat_len}, Dot {pat_dot}")
                    good, bad = self.do_dictionary_gpu(
                        pat_len, pat_dot, hyph_level,
                        good_wt, bad_wt, thresh,
                        left_hyphen_min, right_hyphen_min
                    )
                    print(f"    Added: {good} good, {bad} bad/hopeless")

                if pat_dot == pat_len:
                    break

        self.delete_bad_patterns()

    @property
    def patterns(self):
        """
        Extract all patterns from the trie.
        Returns a list of (pattern_string, value, dot_position) tuples.
        """
        patterns_list = []
        self._collect_patterns(self.trie_root, [], patterns_list)
        return patterns_list

    def _collect_patterns(self, s, prefix, patterns_list):
        """Recursively collect patterns from the trie."""
        for c in range(self.cmin, self.cmax + 1):
            t = s + c
            if t < len(self.trie_c) and self.trie_c[t] == c:
                new_prefix = prefix + [c]

                # Check for ops at this node
                h = int(self.trie_r[t])
                while h > 0:
                    val = self.ops[h]['val']
                    dot = self.ops[h]['dot']
                    if val > 0 and val < MAX_VAL:
                        # Build pattern string
                        pat_str = ''.join(self.xext[ch] if ch < len(self.xext) else '?' for ch in new_prefix)
                        patterns_list.append((pat_str, val, dot))
                    h = self.ops[h]['op']

                # Recurse into children
                child = int(self.trie_l[t])
                if child > 0:
                    self._collect_patterns(child, new_prefix, patterns_list)

    def evaluate(self):
        """
        Evaluate the current patterns against the dictionary.
        Returns (accuracy, list of failures).
        """
        failures = []
        correct = 0
        total = 0

        for idx in range(len(self.words)):
            word_ints = self.words[idx]
            word_len = self.word_lens[idx]
            expected_dots = self.dots[idx]

            # Get predicted hyphenation using current patterns
            predicted = self._hyphenate_word(word_ints[:word_len])

            # Compare
            match = True
            for i in range(1, word_len - 1):  # Skip edge markers
                exp = expected_dots[i]
                pred_val = predicted[i] if i < len(predicted) else 0
                # Check if hyphenation matches
                exp_hyph = (exp == IS_HYF)
                pred_hyph = (pred_val % 2 == 1) and (pred_val > 0)
                if exp_hyph != pred_hyph:
                    match = False
                    break

            if match:
                correct += 1
            else:
                # Build word string for error reporting
                word_str = ''.join(self.xext[c] if c < len(self.xext) else '?' for c in word_ints[:word_len])
                exp_str = self._format_hyphenated(word_ints[:word_len], expected_dots[:word_len], is_expected=True)
                pred_str = self._format_hyphenated(word_ints[:word_len], predicted)
                failures.append((word_str.strip('.'), exp_str, pred_str))

            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, failures

    def _hyphenate_word(self, word_ints):
        """Hyphenate a single word using current patterns."""
        hvals = [0] * len(word_ints)

        for spos in range(len(word_ints) - 1, -1, -1):
            fpos = spos + 1
            if fpos >= len(word_ints):
                continue

            char_code = word_ints[fpos]
            t = self.trie_root + char_code

            while True:
                if t >= len(self.trie_c):
                    break

                if self.trie_c[t] == char_code:
                    h = int(self.trie_r[t])
                    while h > 0:
                        op_val = self.ops[h]['val']
                        op_dot = self.ops[h]['dot']
                        dpos = spos + op_dot

                        if dpos < len(hvals) and op_val < MAX_VAL:
                            if hvals[dpos] < op_val:
                                hvals[dpos] = op_val

                        h = self.ops[h]['op']

                    t = int(self.trie_l[t])
                    if t == 0:
                        break

                    fpos += 1
                    if fpos >= len(word_ints):
                        break
                    char_code = word_ints[fpos]
                    t = t + char_code
                else:
                    break

        return hvals

    def _format_hyphenated(self, word_ints, hvals, is_expected=False):
        """Format a word with hyphenation marks.
        
        Args:
            word_ints: List of character codes
            hvals: Hyphenation values (IS_HYF for expected, or hvals for predicted)
            is_expected: True if hvals contains IS_HYF/NO_HYF constants from dictionary
        """
        result = []
        for i, c in enumerate(word_ints):
            ch = self.xext[c] if c < len(self.xext) else '?'
            if ch != '.':
                result.append(ch)
                # Check if hyphen should appear AFTER this character
                if i < len(hvals):
                    if is_expected:
                        # For expected: check if == IS_HYF (12)
                        if hvals[i] == IS_HYF:
                            result.append('-')
                    else:
                        # For predicted: check if odd and > 0
                        if hvals[i] % 2 == 1 and hvals[i] > 0:
                            result.append('-')
        # Remove trailing hyphen if any
        if result and result[-1] == '-':
            result = result[:-1]
        return ''.join(result)

    def export_patterns(self, filename):
        """Export patterns to a file in TeX hyphenation format."""
        pats = self.patterns
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("% Hyphenation patterns for Indonesian\n")
            f.write("% Generated by OrthosGPU\n")
            f.write("\\patterns{\n")
            for pat_str, val, dot in sorted(pats, key=lambda x: x[0]):
                # Insert the value at the dot position
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
        print(f"Exported {len(pats)} patterns to {filename}")

if __name__ == "__main__":
    pass
