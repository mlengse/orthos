# @title OrthosML - ML-based Pattern Generator
# orthos_ml.py - Neural network approach for hyphenation pattern generation
# Uses PyTorch for GPU acceleration (compatible with Google Colab)

"""
OrthosML - Machine Learning Pattern Generator untuk Pemenggalan Bahasa Indonesia

Alternatif untuk orthos_gpu/cpu yang menggunakan neural network (BiLSTM)
untuk mempelajari pattern hyphenation dari data training.

Features:
- PyTorch-based (GPU support di Google Colab)
- API konsisten dengan OrthosCPU
- Multi-level pattern export (level 1-9)
- Checkpoint support untuk training panjang

Author: Generated for patgen-train-colab
"""

import os
import json
import random
from typing import List, Tuple, Dict, Optional

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. OrthosML cannot be used.")


# ==========================================
# DATASET
# ==========================================

class HyphenDataset(Dataset):
    """
    Load hyphenation data dari .dic file.
    Format: kata-ber-pe-meng-gal-an (satu kata per baris)
    """
    
    def __init__(self, filepath: str = None, max_len: int = 50):
        self.max_len = max_len
        self.words = []
        self.labels = []
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2char = {0: '<PAD>', 1: '<UNK>'}
        
        if filepath:
            self._load_file(filepath)
        
    def _load_file(self, filepath: str):
        print(f"Loading dataset from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '-' not in line:
                    continue
                
                parts = line.split('-')
                chars = ''.join(parts)
                
                label = []
                for i, part in enumerate(parts):
                    for j, c in enumerate(part):
                        if ord(c) not in self.char2idx:
                            idx = len(self.char2idx)
                            self.char2idx[ord(c)] = idx
                            self.idx2char[idx] = c
                        
                        if j == len(part) - 1 and i < len(parts) - 1:
                            label.append(1)
                        else:
                            label.append(0)
                
                if len(chars) <= self.max_len:
                    self.words.append(chars)
                    self.labels.append(label)
        
        print(f"  Loaded {len(self.words)} words, vocab size: {len(self.char2idx)}")
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        label = self.labels[idx]
        
        char_indices = [self.char2idx.get(ord(c), 1) for c in word]
        
        pad_len = self.max_len - len(char_indices)
        char_indices = char_indices + [0] * pad_len
        label = label + [0] * pad_len
        
        return (
            torch.tensor(char_indices, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
            len(word)
        )
    
    def save_vocab(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': {str(k): v for k, v in self.char2idx.items()},
                'idx2char': {str(k): v for k, v in self.idx2char.items()}
            }, f, ensure_ascii=False)


# ==========================================
# MODEL
# ==========================================

class HyphenModel(nn.Module):
    """BiLSTM-based sequence labeling for hyphenation."""
    
    def __init__(
        self, 
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        batch_size, max_len = x.size()
        
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            
            if lstm_out.size(1) < max_len:
                padding = torch.zeros(
                    batch_size, max_len - lstm_out.size(1), lstm_out.size(2),
                    device=lstm_out.device, dtype=lstm_out.dtype
                )
                lstm_out = torch.cat([lstm_out, padding], dim=1)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        out = self.fc(self.dropout(lstm_out))
        out = torch.sigmoid(out.squeeze(-1))
        
        return out


# ==========================================
# ORTHOS ML WRAPPER
# ==========================================

class OrthosML:
    """
    ML-based pattern generator with API compatible with OrthosCPU.
    
    Uses BiLSTM neural network for sequence labeling to predict
    hyphenation positions. Trained on .dic file with hyphenated words.
    
    Usage:
        orthos = OrthosML()
        orthos.load_dictionary_safe('indonesia_training.dic')
        orthos.generate_level(2, 5, 1, 1, 1, 10, 2, 2)
        orthos.export_patterns('hyph-id.tex')
    """
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for OrthosML")
        
        self.checkpoint_dir = checkpoint_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.dataset = None
        self.model = None
        self.trainer = None
        
        # Pattern storage (for API compatibility)
        self._patterns = []
        self._pattern_stats = {}
        
        # Baseline patterns from existing .tex file
        self._baseline_patterns = []
        self._parsed_patterns = {}
        self._failure_indices = None
        
        # Training config
        self.batch_size = 64
        self.epochs_per_level = 5
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"OrthosML initialized on {self.device}")
    
    def load_baseline_patterns(self, filepath: str) -> bool:
        """
        Load baseline patterns from existing .tex file.
        These patterns will be included in the final export.
        
        Args:
            filepath: Path to .tex file with existing patterns
            
        Returns:
            True if successful
        """
        try:
            patterns = []
            in_patterns = False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # Strip inline comments (everything after %)
                    if '%' in line:
                        line = line.split('%')[0]
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Detect \patterns{ block
                    if '\\patterns{' in line:
                        in_patterns = True
                        line = line.split('\\patterns{')[-1]
                    
                    # End of patterns block
                    if '}' in line and in_patterns:
                        line = line.split('}')[0]
                        in_patterns = False
                    
                    if in_patterns or line:
                        # Parse patterns from line
                        for pat in line.split():
                            if pat:
                                patterns.append(pat)
            
            self._baseline_patterns = patterns
            
            # Parse patterns into lookup structure
            self._parsed_patterns = self._parse_patterns_for_lookup(patterns)
            
            print(f"  Loaded {len(patterns)} baseline patterns from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading baseline patterns: {e}")
            return False
    
    def _parse_patterns_for_lookup(self, patterns: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """Parse TeX patterns into a lookup dictionary."""
        parsed = {}
        for pat in patterns:
            if not pat or pat.startswith('%'):
                continue
            
            # Extract letters and numbers
            letters = ''.join(c for c in pat if not c.isdigit())
            if not letters:
                continue
            
            values = []
            pos = 0
            for i, c in enumerate(pat):
                if c.isdigit():
                    values.append((pos, int(c)))
                else:
                    pos += 1
            
            if letters not in parsed:
                parsed[letters] = []
            parsed[letters].extend(values)
        
        return parsed
    
    def hyphenate_word(self, word: str) -> str:
        """
        Hyphenate a word using baseline patterns.
        
        Args:
            word: Word to hyphenate
            
        Returns:
            Hyphenated word (e.g., 'a-ba-di')
        """
        if not self._parsed_patterns:
            return word
        
        word_lower = word.lower()
        padded = '.' + word_lower + '.'
        values = [0] * (len(word_lower) + 1)
        
        # Apply all matching patterns
        for i in range(len(padded)):
            for length in range(1, len(padded) - i + 1):
                substring = padded[i:i+length]
                if substring in self._parsed_patterns:
                    for pos, val in self._parsed_patterns[substring]:
                        word_pos = i + pos - 1
                        if 0 <= word_pos < len(values):
                            values[word_pos] = max(values[word_pos], val)
        
        # Build hyphenated word (odd values = hyphen)
        result = []
        for i, c in enumerate(word_lower):
            if i > 0 and values[i] % 2 == 1:
                result.append('-')
            result.append(c)
        
        return ''.join(result)
    
    def test_with_baseline(self, word: str, expected: str) -> bool:
        """
        Test if a word is correctly hyphenated by baseline patterns.
        
        Args:
            word: Original word (without hyphens)
            expected: Expected hyphenation (e.g., 'a-ba-di')
            
        Returns:
            True if baseline produces correct result
        """
        actual = self.hyphenate_word(word)
        return actual == expected
    
    def filter_training_data(self) -> Tuple[List[int], int, int]:
        """
        Filter training data to only words that FAIL with baseline.
        
        Returns:
            Tuple of (failure_indices, num_passed, num_failed)
        """
        if self.dataset is None:
            print("Error: Load dictionary first")
            return [], 0, 0
        
        if not self._parsed_patterns:
            print("Warning: No baseline patterns loaded, using all data")
            return list(range(len(self.dataset))), 0, len(self.dataset)
        
        passed = 0
        failed = 0
        failure_indices = []
        
        for idx in range(len(self.dataset)):
            word = self.dataset.words[idx]
            label = self.dataset.labels[idx]
            
            # Build expected hyphenation
            expected = ""
            for i, c in enumerate(word):
                expected += c
                if i < len(label) and label[i] == 1:
                    expected += "-"
            
            # Test with baseline
            if self.test_with_baseline(word, expected):
                passed += 1
            else:
                failed += 1
                failure_indices.append(idx)
        
        print(f"  Baseline test: {passed} passed, {failed} failed ({100*failed/(passed+failed):.1f}% failures)")
        
        # Store for later use
        self._failure_indices = failure_indices
        
        return failure_indices, passed, failed
    
    def load_dictionary_safe(self, filepath: str) -> bool:
        """
        Load dictionary file (API compatible with OrthosCPU).
        
        Args:
            filepath: Path to .dic file with hyphenated words
            
        Returns:
            True if successful
        """
        try:
            self.dataset = HyphenDataset(filepath)
            
            # Initialize model
            self.model = HyphenModel(vocab_size=len(self.dataset.char2idx))
            self.model.to(self.device)
            
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            return True
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return False
    
    def generate_level(
        self, 
        pat_start: int, 
        pat_finish: int, 
        hyph_level: int,
        good_wt: int = 1, 
        bad_wt: int = 1, 
        thresh: int = 10,
        left_hyphen_min: int = 2, 
        right_hyphen_min: int = 2
    ):
        """
        Generate patterns for a specific level (API compatible with OrthosCPU).
        
        For ML approach, this trains the model for more epochs.
        The hyph_level affects how patterns are exported later.
        
        Args:
            pat_start: Minimum pattern length (used for filtering)
            pat_finish: Maximum pattern length (used for filtering)
            hyph_level: Hyphenation level (1,3,5,7 = allow; 2,4,6,8 = inhibit)
            good_wt: Weight for correct hyphenation
            bad_wt: Weight for incorrect hyphenation
            thresh: Minimum threshold for pattern inclusion
            left_hyphen_min: Minimum chars before hyphen
            right_hyphen_min: Minimum chars after hyphen
        """
        if self.dataset is None or self.model is None:
            print("Error: Load dictionary first")
            return
        
        print(f"Training for level {hyph_level} (pat_len {pat_start}-{pat_finish})...")
        
        # Store config for export
        self._current_level = hyph_level
        self._pat_range = (pat_start, pat_finish)
        self._thresh = thresh
        self._margins = (left_hyphen_min, right_hyphen_min)
        
        # Use filtered failures if available, otherwise full dataset
        if self._failure_indices is not None and len(self._failure_indices) > 0:
            print(f"  Training on {len(self._failure_indices)} failures (filtered)")
            training_indices = self._failure_indices
            subset = torch.utils.data.Subset(self.dataset, training_indices)
        else:
            print(f"  Training on full dataset ({len(self.dataset)} words)")
            subset = self.dataset
        
        # Setup training
        train_size = int(0.9 * len(subset))
        val_size = len(subset) - train_size
        
        if train_size < 10:
            print(f"  Warning: Very small training set ({train_size}), using all data")
            train_dataset = subset
        else:
            generator = torch.Generator().manual_seed(42)
            train_dataset, _ = torch.utils.data.random_split(
                subset, [train_size, val_size], generator=generator
            )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss(reduction='none')
        
        for epoch in range(self.epochs_per_level):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_positions = 0
            
            for batch in train_loader:
                chars, labels, lengths = batch
                chars = chars.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(chars, lengths)
                
                mask = torch.arange(chars.size(1), device=self.device).expand(
                    len(lengths), -1
                ) < lengths.unsqueeze(1)
                
                loss = criterion(outputs, labels)
                loss = (loss * mask).sum() / mask.sum()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                total_correct += ((preds == labels) * mask).sum().item()
                total_positions += mask.sum().item()
            
            accuracy = total_correct / total_positions
            print(f"  Epoch {epoch+1}/{self.epochs_per_level}: "
                  f"loss={total_loss/len(train_loader):.4f}, acc={accuracy:.2%}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'level_{hyph_level}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'level': hyph_level,
            'pat_range': (pat_start, pat_finish),
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def export_patterns(self, filename: str, 
                       threshold: float = 0.90,
                       min_word_coverage: int = 2,
                       max_pattern_len: int = 5):
        """
        Export patterns to TeX format (API compatible with OrthosCPU).
        
        Uses multi-level pattern generation based on confidence:
        - Level 1, 3, 5, 7: Allow hyphen (confidence thresholds)
        - Level 2, 4, 6, 8: Inhibit hyphen (confidence thresholds)
        
        Args:
            filename: Output .tex file path
            threshold: Base threshold for pattern extraction
            min_word_coverage: Minimum words affected by pattern
            max_pattern_len: Maximum pattern length
        """
        if self.dataset is None or self.model is None:
            print("Error: Train model first")
            return 0, 0
        
        print(f"Exporting patterns to {filename}...")
        
        self.model.eval()
        
        # Collect pattern statistics with confidence levels
        pattern_stats: Dict[str, Dict] = {}
        exception_words = set()
        
        with torch.no_grad():
            loader = DataLoader(self.dataset, batch_size=64, shuffle=False)
            batch_offset = 0
            
            for batch in loader:
                chars, labels, lengths = batch
                chars = chars.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(chars, lengths)
                outputs = outputs.cpu().numpy()
                chars_np = chars.cpu().numpy()
                lengths_np = lengths.cpu().numpy()
                labels_np = labels.numpy()
                
                for i in range(len(chars_np)):
                    word_idx = batch_offset + i
                    word_len = lengths_np[i]
                    
                    # Check word accuracy
                    word_correct = True
                    for pos in range(word_len - 1):
                        pred = outputs[i, pos] > 0.5
                        expected = labels_np[i, pos] > 0.5
                        if pred != expected:
                            word_correct = False
                            break
                    
                    # Collect patterns
                    for pos in range(word_len - 1):
                        prob = outputs[i, pos]
                        
                        for ngram_len in range(2, max_pattern_len + 1):
                            start = max(0, pos - ngram_len + 2)
                            end = min(word_len, pos + ngram_len)
                            
                            if end - start < 2 or end - start > max_pattern_len:
                                continue
                            
                            chars_slice = chars_np[i, start:end]
                            pattern = ''.join(
                                self.dataset.idx2char.get(int(c), '?') 
                                for c in chars_slice
                            )
                            
                            if '<' in pattern or '?' in pattern:
                                continue
                            
                            dot = pos - start + 1
                            key = f"{pattern}:{dot}"
                            
                            if key not in pattern_stats:
                                pattern_stats[key] = {
                                    'pattern': pattern, 
                                    'dot': dot,
                                    'probs': [],
                                    'words': set()
                                }
                            
                            pattern_stats[key]['probs'].append(prob)
                            pattern_stats[key]['words'].add(word_idx)
                    
                    if not word_correct:
                        exception_words.add(word_idx)
                
                batch_offset += len(chars_np)
        
        # Generate multi-level patterns
        general_patterns = []
        
        for key, stats in pattern_stats.items():
            coverage = len(stats['words'])
            probs = stats['probs']
            
            if coverage < min_word_coverage:
                continue
            
            if len(probs) < 5:
                continue
            
            avg_prob = sum(probs) / len(probs)
            
            # Multi-level based on confidence
            # Baseline uses levels 1-2, ML uses levels 3-9 for patching
            # Level 3, 5, 7, 9 = allow (odd) - higher = more confident
            # Level 4, 6, 8 = inhibit (even) - higher = more confident
            # ML levels must be > 2 to override baseline when needed
            level = None
            
            if avg_prob > 0.95:
                level = 9  # Very high confidence allow
            elif avg_prob > 0.85:
                level = 7  # High confidence allow
            elif avg_prob > 0.70:
                level = 5  # Medium-high confidence allow
            elif avg_prob > 0.50:
                level = 3  # Medium confidence allow (lowest ML allow)
            elif avg_prob < 0.05:
                level = 8  # Very high confidence inhibit
            elif avg_prob < 0.15:
                level = 6  # High confidence inhibit
            elif avg_prob < 0.30:
                level = 4  # Medium confidence inhibit (lowest ML inhibit)
            # Skip level 2 - reserved for baseline inhibit
            
            if level is not None:
                pattern = stats['pattern']
                dot = stats['dot']
                if 0 < dot < len(pattern):
                    tex_pattern = pattern[:dot] + str(level) + pattern[dot:]
                    general_patterns.append((tex_pattern, level, avg_prob))
        
        # Sort by level priority (higher levels first for same pattern)
        general_patterns.sort(key=lambda x: (-x[1], x[0]))
        
        # Remove duplicates keeping highest level
        seen_patterns = set()
        unique_patterns = []
        for pat, level, prob in general_patterns:
            base = ''.join(c for c in pat if not c.isdigit())
            if base not in seen_patterns:
                seen_patterns.add(base)
                unique_patterns.append(pat)
        
        # Build exception list
        exception_list = []
        for word_idx in exception_words:
            if word_idx < len(self.dataset.words):
                word = self.dataset.words[word_idx]
                label = self.dataset.labels[word_idx]
                hyphenated = ""
                for i, c in enumerate(word):
                    hyphenated += c
                    if i < len(label) and label[i] == 1:
                        hyphenated += "-"
                exception_list.append(hyphenated)
        
        exception_list = sorted(set(exception_list))[:2000]
        
        # Merge baseline patterns with ML patterns
        # Baseline patterns take priority (they are rule-based EYD)
        all_patterns = list(self._baseline_patterns)  # Start with baseline
        
        # Track which pattern bases are already covered by baseline
        baseline_bases = set()
        for pat in all_patterns:
            base = ''.join(c for c in pat if not c.isdigit())
            baseline_bases.add(base)
        
        # Add ML patterns that don't conflict with baseline
        for pat in unique_patterns:
            base = ''.join(c for c in pat if not c.isdigit())
            if base not in baseline_bases:
                all_patterns.append(pat)
        
        # Write TeX file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("% Hyphenation patterns generated by OrthosML\n")
            f.write(f"% Baseline: {len(self._baseline_patterns)}, ML: {len(all_patterns) - len(self._baseline_patterns)}\n")
            f.write(f"% Total patterns: {len(all_patterns)}, Exceptions: {len(exception_list)}\n")
            f.write("% Uses multi-level patterns (1-9) per Liang thesis\n\n")
            
            f.write("\\patterns{\n")
            f.write("% === BASELINE (EYD V Rules) ===\n")
            for i, pat in enumerate(self._baseline_patterns):
                f.write(pat)
                if (i + 1) % 10 == 0:
                    f.write("\n")
                else:
                    f.write(" ")
            
            f.write("\n% === ML-GENERATED ===\n")
            ml_patterns = all_patterns[len(self._baseline_patterns):]
            ml_patterns.sort()
            for i, pat in enumerate(ml_patterns):
                f.write(pat)
                if (i + 1) % 10 == 0:
                    f.write("\n")
                else:
                    f.write(" ")
            f.write("\n}\n\n")
            
            if exception_list:
                f.write("\\hyphenation{\n")
                for word in exception_list:
                    f.write(f"  {word}\n")
                f.write("}\n")
        
        print(f"  Exported {len(all_patterns)} patterns (baseline: {len(self._baseline_patterns)}, ML: {len(ml_patterns)}) + {len(exception_list)} exceptions")
        return len(all_patterns), len(exception_list)
    
    @property
    def patterns(self) -> List[Tuple[str, int, int]]:
        """Get list of patterns (API compatible with OrthosCPU)."""
        return self._patterns


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================

def is_available() -> bool:
    """Check if OrthosML can be used."""
    return PYTORCH_AVAILABLE


def create_orthos_ml(checkpoint_dir: str = './checkpoints') -> OrthosML:
    """Factory function to create OrthosML instance."""
    return OrthosML(checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    print(f"PyTorch available: {PYTORCH_AVAILABLE}")
    if PYTORCH_AVAILABLE:
        print(f"CUDA available: {torch.cuda.is_available()}")
        orthos = OrthosML()
        print(f"OrthosML ready on device: {orthos.device}")
