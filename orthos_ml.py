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
        
        # Training config
        self.batch_size = 64
        self.epochs_per_level = 5
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"OrthosML initialized on {self.device}")
    
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
        
        # Setup training
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=generator
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
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
            # Level 1, 3, 5, 7, 9 = allow (odd) - higher = more confident
            # Level 2, 4, 6, 8 = inhibit (even) - higher = more confident
            # IMPORTANT: Allow levels must be >= inhibit levels to win
            level = None
            
            if avg_prob > 0.95:
                level = 9  # Very high confidence allow (highest priority)
            elif avg_prob > 0.85:
                level = 7  # High confidence allow
            elif avg_prob > 0.70:
                level = 5  # Medium-high confidence allow
            elif avg_prob > 0.50:
                level = 3  # Medium confidence allow
            elif avg_prob < 0.05:
                level = 8  # Very high confidence inhibit (but < 9)
            elif avg_prob < 0.15:
                level = 6  # High confidence inhibit (but < 7)
            elif avg_prob < 0.30:
                level = 4  # Medium-high confidence inhibit (but < 5)
            elif avg_prob < 0.50:
                level = 2  # Medium confidence inhibit (but < 3)
            
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
        
        # Write TeX file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("% Hyphenation patterns generated by OrthosML\n")
            f.write(f"% Patterns: {len(unique_patterns)}, Exceptions: {len(exception_list)}\n")
            f.write("% Uses multi-level patterns (1-9) per Liang thesis\n\n")
            
            f.write("\\patterns{\n")
            unique_patterns.sort()
            for i, pat in enumerate(unique_patterns):
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
        
        print(f"  Exported {len(unique_patterns)} patterns + {len(exception_list)} exceptions")
        return len(unique_patterns), len(exception_list)
    
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
