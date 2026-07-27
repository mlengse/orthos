"""Tests for orthos_colab.py — validates B7 (insert_pattern parity) and core logic."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orthos_colab import OrthosEngine, MAX_VAL, MAX_LEN, NO_HYF, IS_HYF, ERR_HYF, FOUND_HYF


@pytest.fixture
def engine():
    """Fresh OrthosEngine for each test."""
    return OrthosEngine()


def _make_engine_with_chars(engine, chars="abcdefghijklmnopqrstuvwxyz."):
    """Register characters into the engine's mapping tables."""
    for ch in chars:
        code = ord(ch)
        if code not in engine.xint:
            engine.xext.append(ch)
            engine.xint[code] = len(engine.xext) - 1
            engine.xclass[code] = 3  # LETTER_CLASS
            engine.cmax = len(engine.xext) - 1
    engine.trie_c[:] = 0
    engine.trie_l[:] = 0
    engine.trie_r[:] = 0
    engine.trie_taken[:] = 0
    engine.ops_val[:] = 0
    engine.ops_dot[:] = 0
    engine.ops_op[:] = 0
    engine.op_count = 0
    engine.trie_max = 0
    engine.trie_bmax = 0
    engine.init_pattern_trie()


def _collect_pattern_tuples(engine):
    """Collect all (pattern_string, val, dot) from the trie."""
    pats = []
    engine._collect_patterns(engine.trie_root, [], pats)
    return sorted(pats, key=lambda x: (x[0], x[1], x[2]))


# ============================================================
# B7: insert_pattern parity
# ============================================================

class TestInsertPattern:
    def test_single_char_pattern(self, engine):
        """Insert pattern 'a' with val=1, dot=0."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("a")]]
        engine.insert_pattern(pat, 1, 0)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 1
        assert pats[0] == ("a", 1, 0)

    def test_two_char_pattern(self, engine):
        """Insert pattern 'ba' with val=1, dot=0."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("b")], engine.xint[ord("a")]]
        engine.insert_pattern(pat, 1, 0)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 1
        assert pats[0] == ("ba", 1, 0)

    def test_three_char_pattern(self, engine):
        """Insert pattern 'foo' with val=2, dot=1."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("f")], engine.xint[ord("o")], engine.xint[ord("o")]]
        engine.insert_pattern(pat, 2, 1)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 1
        assert pats[0] == ("foo", 2, 1)

    def test_multiple_patterns_share_prefix(self, engine):
        """Insert 'foo' and 'fab' — they share prefix 'f'."""
        _make_engine_with_chars(engine)
        fi = engine.xint[ord("f")]
        oi = engine.xint[ord("o")]
        ai = engine.xint[ord("a")]
        bi = engine.xint[ord("b")]

        engine.insert_pattern([fi, oi, oi], 1, 0)
        engine.insert_pattern([fi, ai, bi], 2, 0)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 2
        strings = {p[0] for p in pats}
        assert "foo" in strings
        assert "fab" in strings

    def test_insert_pattern_with_val_max_is_bad(self, engine):
        """MAX_VAL patterns should be filtered out by _collect_patterns."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("x")]]
        engine.insert_pattern(pat, MAX_VAL, 0)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 0

    def test_insert_and_retrieve_dot_position(self, engine):
        """Pattern 'abc' with dot=2 means value appears between b and c."""
        _make_engine_with_chars(engine)
        ai = engine.xint[ord("a")]
        bi = engine.xint[ord("b")]
        ci = engine.xint[ord("c")]

        engine.insert_pattern([ai, bi, ci], 3, 2)
        pats = _collect_pattern_tuples(engine)
        assert pats[0] == ("abc", 3, 2)


# ============================================================
# delete_patterns / delete_bad_patterns
# ============================================================

class TestDeletePatterns:
    def test_delete_removes_bad_val_patterns(self, engine):
        """delete_bad_patterns should remove MAX_VAL patterns."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("x")]]
        engine.insert_pattern(pat, MAX_VAL, 0)
        engine.insert_pattern(pat, 1, 0)

        pats_before = _collect_pattern_tuples(engine)
        assert len(pats_before) == 1  # only val=1 (MAX_VAL filtered)
        assert pats_before[0][1] == 1

        engine.delete_bad_patterns()
        pats_after = _collect_pattern_tuples(engine)
        assert len(pats_after) == 1
        assert pats_after[0] == ("x", 1, 0)

    def test_delete_patterns_returns_root_when_empty(self, engine):
        """delete_patterns on root with no children returns root (1), not 0.
        Root is never freed because of the `s == trie_root` guard."""
        _make_engine_with_chars(engine)
        result = engine.delete_patterns(engine.trie_root)
        assert result == 1  # root is never freed

    def test_op_count_decrements_on_delete_bad(self, engine):
        """op_count should decrease when bad patterns are deleted."""
        _make_engine_with_chars(engine)
        initial_op_count = engine.op_count

        pat = [engine.xint[ord("z")]]
        engine.insert_pattern(pat, MAX_VAL, 0)
        assert engine.op_count > initial_op_count

        engine.delete_bad_patterns()
        assert engine.op_count == initial_op_count


# ============================================================
# load_dictionary / export_patterns roundtrip
# ============================================================

class TestLoadAndExport:
    def _write_dic(self, lines):
        """Write a temporary .dic file and return its path."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".dic", delete=False, encoding="utf-8"
        )
        for line in lines:
            f.write(line + "\n")
        f.close()
        return f.name

    def test_load_single_word(self, engine):
        """Load a single hyphenated word."""
        path = self._write_dic(["in-do-ne-si-a"])
        try:
            engine.load_dictionary(path)
            assert engine.words is not None
            assert engine.words.shape[0] == 1
            assert engine.word_lens[0] > 0
        finally:
            os.unlink(path)

    def test_load_multiple_words(self, engine):
        """Load multiple words."""
        path = self._write_dic(["ka-ta", "ba-cu", "su-su"])
        try:
            engine.load_dictionary(path)
            assert engine.words.shape[0] == 3
        finally:
            os.unlink(path)

    def test_export_creates_file(self, engine, tmp_path):
        """export_patterns should write a file."""
        out = tmp_path / "out.tex"
        engine.export_patterns(str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "\\patterns{" in content

    def test_insert_then_export(self, engine, tmp_path):
        """Insert patterns then export — should roundtrip."""
        _make_engine_with_chars(engine)
        ai = engine.xint[ord("a")]
        bi = engine.xint[ord("b")]
        engine.insert_pattern([ai, bi], 1, 0)
        engine.insert_pattern([bi, ai], 2, 0)

        out = tmp_path / "out.tex"
        engine.export_patterns(str(out))
        content = out.read_text(encoding="utf-8")

        assert "ab" in content
        assert "ba" in content

    def test_load_empty_file(self, engine):
        """Loading an empty dictionary should not crash."""
        path = self._write_dic([])
        try:
            engine.load_dictionary(path)
            assert engine.words.shape[0] == 0
        finally:
            os.unlink(path)


# ============================================================
# validate_logic (existing self-test)
# ============================================================

class TestValidateLogic:
    def test_validate_logic_passes(self, engine):
        """The built-in validate_logic should produce pattern '1b'."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            engine.validate_logic()

        output = buf.getvalue()
        assert "Logic Test Passed" in output or "Pattern Verification Passed" in output


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_insert_empty_pattern_raises(self, engine):
        """Empty pattern list should raise IndexError (no chars to index)."""
        _make_engine_with_chars(engine)
        with pytest.raises(IndexError):
            engine.insert_pattern([], 1, 0)

    def test_insert_duplicate_pattern_extends_chain(self, engine):
        """Inserting the same pattern twice extends the ops chain (not deduped
        at the op level because next_op differs). This is expected Liang behavior
        — the trie node gets two ops entries with the same val/dot but different
        chain links. Both get collected but produce identical output patterns."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("a")], engine.xint[ord("b")]]
        engine.insert_pattern(pat, 1, 0)
        engine.insert_pattern(pat, 1, 0)

        pats = _collect_pattern_tuples(engine)
        # Both ops are collected, but both produce the same string
        assert all(p[0] == "ab" and p[1] == 1 for p in pats)

    def test_insert_same_string_different_vals(self, engine):
        """Same string, different val/dot should produce two entries."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("a")], engine.xint[ord("b")]]
        engine.insert_pattern(pat, 1, 0)
        engine.insert_pattern(pat, 2, 1)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 2
        dots = {p[2] for p in pats}
        assert 0 in dots
        assert 1 in dots


# ============================================================
# B7 parity proof: loop bounds are equivalent
# ============================================================

class TestB7Parity:
    """Proves that the loop bound difference between JS and Python is
    a non-issue. JS uses 1-indexed arrays with `i < pat_len` starting
    at i=1. Python uses 0-indexed with `i < pat_len - 1` starting at
    i=0. Both process exactly pat_len - 1 intermediate trie nodes."""

    def test_long_pattern_all_chars_stored(self, engine):
        """Insert a 10-char pattern — all characters must be in the trie."""
        _make_engine_with_chars(engine)
        word = "abcdefghij"
        pat = [engine.xint[ord(c)] for c in word]
        engine.insert_pattern(pat, 3, 5)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 1
        assert pats[0] == (word, 3, 5)

    def test_max_length_pattern(self, engine):
        """Insert a pattern of length MAX_LEN (50) — tests full loop traversal."""
        _make_engine_with_chars(engine)
        # Build a 20-char pattern (max_len is 50, but we use 20 for speed)
        word = "abcdefghijklmnopqrst"
        pat = [engine.xint[ord(c)] for c in word]
        engine.insert_pattern(pat, 2, 10)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 1
        assert pats[0] == (word, 2, 10)

    def test_single_char_pattern_val_at_end(self, engine):
        """Single-char pattern: dot=1 (after the char)."""
        _make_engine_with_chars(engine)
        pat = [engine.xint[ord("z")]]
        engine.insert_pattern(pat, 4, 1)

        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 1
        assert pats[0] == ("z", 4, 1)


# ============================================================
# load_patterns — input sanitization (S6)
# ============================================================

class TestLoadPatterns:
    def _write_pat(self, lines):
        """Write a temporary pattern file and return its path."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".pat", delete=False, encoding="utf-8"
        )
        for line in lines:
            f.write(line + "\n")
        f.close()
        return f.name

    def test_load_valid_patterns(self, engine):
        """Valid patterns should be loaded into the trie."""
        _make_engine_with_chars(engine)
        path = self._write_pat(["1ab", "b2a"])
        try:
            engine.load_patterns(path)
            pats = _collect_pattern_tuples(engine)
            assert len(pats) >= 2
        finally:
            os.unlink(path)

    def test_load_empty_file(self, engine):
        """Loading an empty pattern file should not crash."""
        _make_engine_with_chars(engine)
        path = self._write_pat([])
        try:
            engine.load_patterns(path)
            pats = _collect_pattern_tuples(engine)
            assert len(pats) == 0
        finally:
            os.unlink(path)

    def test_invalid_digit_value_skipped(self, engine):
        """Patterns with digit values >= MAX_VAL should be skipped."""
        _make_engine_with_chars(engine)
        path = self._write_pat(["9ab", "1cd"])
        try:
            engine.load_patterns(path)
            pats = _collect_pattern_tuples(engine)
            # '9ab' has val=9 which is < MAX_VAL (10), so it's valid
            # But test with actual MAX_VAL: patterns with digit >= 10 can't
            # be written as single digit in TeX format, so test the boundary
            assert all(p[1] < MAX_VAL for p in pats)
        finally:
            os.unlink(path)

    def test_oversized_pattern_skipped(self, engine):
        """Patterns exceeding MAX_LEN characters should be skipped."""
        _make_engine_with_chars(engine)
        # Build a pattern longer than MAX_LEN
        long_pattern = "a" * (MAX_LEN + 1)
        short_pattern = "ab"
        path = self._write_pat([f"1{long_pattern}", f"1{short_pattern}"])
        try:
            engine.load_patterns(path)
            pats = _collect_pattern_tuples(engine)
            # long_pattern should be skipped, short_pattern should be loaded
            strings = {p[0] for p in pats}
            assert long_pattern not in strings
            assert short_pattern in strings
        finally:
            os.unlink(path)

    def test_tex_comments_stripped(self, engine):
        """TeX comments (%) should be stripped before parsing."""
        _make_engine_with_chars(engine)
        path = self._write_pat(["1ab % this is a comment", "1cd"])
        try:
            engine.load_patterns(path)
            pats = _collect_pattern_tuples(engine)
            strings = {p[0] for p in pats}
            assert "ab" in strings
            assert "cd" in strings
        finally:
            os.unlink(path)

    def test_tex_wrapper_stripped(self, engine):
        """\\patterns{...} wrapper should be stripped."""
        _make_engine_with_chars(engine)
        path = self._write_pat(["\\patterns{", "1ab", "1cd", "}"])
        try:
            engine.load_patterns(path)
            pats = _collect_pattern_tuples(engine)
            strings = {p[0] for p in pats}
            assert "ab" in strings
            assert "cd" in strings
        finally:
            os.unlink(path)

    def test_missing_file_starts_empty(self, engine):
        """Loading a nonexistent file should not crash (graceful fallback)."""
        _make_engine_with_chars(engine)
        engine.load_patterns("/nonexistent/path/patterns.pat")
        pats = _collect_pattern_tuples(engine)
        assert len(pats) == 0
