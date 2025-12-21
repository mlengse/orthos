# @title CPU Implementation (No GPU Required)
# Versi CPU dari orthos_gpu.py untuk environment tanpa GPU

import numpy as np
import time

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


class OrthosCPU:
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
        self.init_pattern_trie()

        # Data
        self.words = None
        self.word_lens = None
        self.dots = None
        self.dotw = None

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
            s = t - self.trieq_c[1]
            if s > TRIE_SIZE - len(self.xext):
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

                pad_len = MAX_LEN - len(word_ints)
                if pad_len > 0:
                    word_ints.extend([0] * pad_len)
                    dots.extend([0] * pad_len)
                    dotw.extend([0] * pad_len)

                words_list.append(word_ints)
                dots_list.append(dots)
                dotw_list.append(dotw)

        self.words = np.array(words_list, dtype=np.uint32)
        self.dots = np.array(dots_list, dtype=np.uint32)
        self.dotw = np.array(dotw_list, dtype=np.uint32)
        self.word_lens = np.array(lens_list, dtype=np.int32)

        print(f"Loaded {len(words_list)} words.")

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

    def _hyphenate_word(self, idx, hvals, no_more, pat_len, pat_dot, hyph_level):
        """CPU implementation of kernel_hyphenate for a single word."""
        wlen = self.word_lens[idx]
        
        # Iterate backwards through the word
        for spos in range(wlen - 1, -1, -1):
            fpos = spos + 1
            if fpos >= MAX_LEN or fpos >= wlen:
                continue
            
            char_code = self.words[idx, fpos]
            t = self.trie_root + char_code
            
            # Traverse Trie
            while True:
                if t >= len(self.trie_c):
                    break
                
                if self.trie_c[t] == char_code:
                    h = self.trie_r[t]
                    while h > 0:
                        op_val = self.ops[h]['val']
                        op_dot = self.ops[h]['dot']
                        op_next = self.ops[h]['op']
                        
                        dpos = spos + op_dot
                        
                        if dpos < MAX_LEN:
                            # Update hvals
                            if op_val < MAX_VAL:
                                if hvals[idx, dpos] < op_val:
                                    hvals[idx, dpos] = op_val
                            
                            # Update no_more
                            if op_val >= hyph_level:
                                cond1 = (fpos - pat_len) <= (dpos - pat_dot)
                                cond2 = (dpos - pat_dot) <= spos
                                if cond1 and cond2:
                                    no_more[idx, dpos] = 1
                        
                        h = op_next
                    
                    t = self.trie_l[t]
                    if t == 0:
                        break
                    
                    fpos += 1
                    if fpos >= wlen:
                        break
                    char_code = self.words[idx, fpos]
                    t = t + char_code
                else:
                    break

    def _extract_candidates(self, idx, hvals, no_more, pat_len, pat_dot, 
                           good_dot, bad_dot, dot_min, dot_max,
                           out_patterns, out_weights, out_mask):
        """CPU implementation of kernel_extract_candidates for a single word."""
        wlen = self.word_lens[idx]
        
        # dpos loop: wlen - dot_max down to dot_min
        for dpos in range(wlen - dot_max, dot_min - 1, -1):
            if no_more[idx, dpos] == 0:
                current_dot = self.dots[idx, dpos]
                is_good = (current_dot == good_dot)
                is_bad = (current_dot == bad_dot)
                
                if is_good or is_bad:
                    spos = dpos - pat_dot
                    out_idx = idx * MAX_LEN + dpos
                    
                    # Check bounds
                    if spos + 1 >= 0 and spos + 1 + pat_len <= wlen:
                        for k in range(pat_len):
                            out_patterns[out_idx, k] = self.words[idx, spos + 1 + k]
                        
                        weight = self.dotw[idx, dpos]
                        
                        if is_good:
                            out_weights[out_idx, 0] = weight
                            out_weights[out_idx, 1] = 0.0
                        else:
                            out_weights[out_idx, 0] = 0.0
                            out_weights[out_idx, 1] = float(weight)
                        
                        out_mask[out_idx] = 1

    def do_dictionary_cpu(self, pat_len, pat_dot, hyph_level, good_wt, bad_wt, thresh, left_hyphen_min, right_hyphen_min):
        """CPU implementation of do_dictionary_gpu."""
        n_words = self.words.shape[0]
        
        # Initialize arrays
        hvals = np.zeros((n_words, MAX_LEN), dtype=np.uint8)
        no_more = np.zeros((n_words, MAX_LEN), dtype=np.uint8)
        
        # Hyphenate all words
        print(f"    Processing {n_words} words on CPU...")
        start_time = time.time()
        
        for idx in range(n_words):
            self._hyphenate_word(idx, hvals, no_more, pat_len, pat_dot, hyph_level)
            
            # Progress indicator
            if (idx + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (n_words - idx - 1) / rate
                print(f"      Progress: {idx + 1}/{n_words} ({100*(idx+1)/n_words:.1f}%) - ETA: {remaining:.1f}s")
        
        # Determine Good/Bad Dot
        if (hyph_level % 2) == 1:
            good_dot = IS_HYF
            bad_dot = NO_HYF
        else:
            good_dot = ERR_HYF
            bad_dot = FOUND_HYF

        dot_min = pat_dot
        if dot_min < (left_hyphen_min + 1):
            dot_min = left_hyphen_min + 1
        dot_max = pat_len - pat_dot
        if dot_max < (right_hyphen_min + 1):
            dot_max = right_hyphen_min + 1

        # Extract candidates
        out_patterns = np.zeros((n_words * MAX_LEN, pat_len), dtype=np.uint32)
        out_weights = np.zeros((n_words * MAX_LEN, 2), dtype=np.float32)
        out_mask = np.zeros((n_words * MAX_LEN), dtype=np.uint8)
        
        for idx in range(n_words):
            self._extract_candidates(idx, hvals, no_more, pat_len, pat_dot,
                                    good_dot, bad_dot, dot_min, dot_max,
                                    out_patterns, out_weights, out_mask)
        
        # Aggregation
        mask = out_mask.astype(bool)
        valid_patterns = out_patterns[mask]
        valid_weights = out_weights[mask]
        
        if valid_patterns.shape[0] == 0:
            return 0, 0
        
        # Sort and aggregate (CPU version using numpy)
        keys = tuple(valid_patterns[:, i] for i in range(pat_len - 1, -1, -1))
        sorted_indices = np.lexsort(keys)
        
        sorted_patterns = valid_patterns[sorted_indices]
        sorted_weights = valid_weights[sorted_indices]
        
        diff = sorted_patterns[1:] != sorted_patterns[:-1]
        boundaries = np.any(diff, axis=1)
        
        run_ids = np.zeros(len(sorted_patterns), dtype=np.int32)
        run_ids[1:] = np.cumsum(boundaries)
        
        num_unique = int(run_ids[-1]) + 1
        unique_weights = np.zeros((num_unique, 2), dtype=np.float32)
        
        np.add.at(unique_weights, run_ids, sorted_weights)
        
        # Get unique patterns
        change_points = np.flatnonzero(boundaries) + 1
        unique_indices = np.concatenate((np.array([0]), change_points))
        unique_patterns_arr = sorted_patterns[unique_indices]
        
        # Selection & Update
        good_pat_count = 0
        bad_pat_count = 0
        
        for i in range(num_unique):
            w_good = unique_weights[i, 0]
            w_bad = unique_weights[i, 1]
            
            pat = unique_patterns_arr[i].tolist()
            
            if (good_wt * w_good) < thresh:
                self.insert_pattern(pat, MAX_VAL, pat_dot)
                bad_pat_count += 1
            elif (good_wt * w_good - bad_wt * w_bad) >= thresh:
                self.insert_pattern(pat, hyph_level, pat_dot)
                good_pat_count += 1
        
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
                    good, bad = self.do_dictionary_cpu(
                        pat_len, pat_dot, hyph_level,
                        good_wt, bad_wt, thresh,
                        left_hyphen_min, right_hyphen_min
                    )
                    print(f"    Added: {good} good, {bad} bad/hopeless")

                if pat_dot == pat_len:
                    break

        self.delete_bad_patterns()


if __name__ == "__main__":
    pass
