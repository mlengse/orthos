"""Cross-implementation integration tests.

Safety net for refactors (see audit item 6.5): JS and Python must produce
identical pattern output from the same input, and both must match a committed
golden snapshot (so a shared regression can't silently keep both "in sync").
"""

import os
import sys
import tempfile
import subprocess
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GOLDEN_FILE = os.path.join(os.path.dirname(__file__), "golden_expected.txt")

SMALL_DIC = "ka-ta\nba-cu\nsu-su\n"
SMALL_INPUTS = "1 1\n1 1\n1 1\n1 1 1\nn\nn"

# Real Indonesian hyphenation from KBBI (all lowercase for deterministic parity).
REAL_DIC = "\n".join([
    "in-do-ne-si-a",
    "pe-me-rin-tah",
    "pen-di-dik-an",
    "ma-sya-ra-kat",
    "pe-nge-ta-hu-an",
    "per-pus-ta-ka-an",
    "u-ni-ver-si-tas",
    "tek-no-lo-gi",
    "in-for-ma-si",
    "ko-mu-ni-ka-si",
    "trans-por-ta-si",
    "ad-mi-nis-tra-si",
    "or-ga-ni-sa-si",
    "im-ple-men-ta-si",
    "pro-duk-si",
    "dis-tri-bu-si",
    "ka-rak-te-ris-tik",
    "do-ku-men-ta-si",
    "in-fra-struk-tur",
    "ke-se-jah-te-ra-an",
    "ke-se-hat-an",
    "ku-a-li-tas",
    "per-kem-bang-an",
    "ke-bu-tuh-an",
    "per-u-sa-ha-an",
    "ling-kung-an",
    "pe-ngem-bang-an",
    "ke-u-ang-an",
    "wi-la-yah",
    "ber-ba-gai",
]) + "\n"

# left/right, hyph 1..2 (two levels), then per level: pat 1..2 + good/bad/thresh.
# Trailing "n" answers the JS-only "hyphenate word list?" prompt.
MULTI_INPUTS = (
    "1 1\n"
    "1 2\n"
    "1 2\n1 1 1\n"
    "1 2\n1 1 1\n"
    "n"
)


def run_js(dic_path, pat_in_path, pat_out_path, inputs):
    proc = subprocess.Popen(
        ["node", "orthos.js", dic_path, pat_in_path, pat_out_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=os.path.join(os.path.dirname(__file__), "..")
    )
    for line in inputs.split("\n"):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        time.sleep(0.4)
    stdout, stderr = proc.communicate(timeout=120)
    return proc.returncode, stdout, stderr


def run_python(dic_path, pat_in_path, pat_out_path, inputs):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "orthos_colab.py", dic_path, pat_in_path, pat_out_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
        cwd=os.path.join(os.path.dirname(__file__), ".."), env=env
    )
    stdout, stderr = proc.communicate(input=inputs, timeout=120)
    return proc.returncode, stdout, stderr


def normalize_patterns(content):
    lines = content.splitlines()
    pat_lines = []
    inside = False
    has_tex = any("\\patterns" in line for line in lines)
    for line in lines:
        stripped = line.strip()
        if has_tex:
            if "\\patterns" in stripped:
                inside = True
                continue
            if inside and stripped == "}":
                break
            if inside and stripped and not stripped.startswith("%"):
                pat_lines.append(stripped)
        else:
            if stripped and not stripped.startswith("%"):
                pat_lines.append(stripped)
    return sorted(pat_lines)


def write_manifest(tmp, dic, pat_in):
    dic_path = os.path.join(tmp, "test.dic")
    pat_in_path = os.path.join(tmp, "in.pat")
    with open(dic_path, "w", encoding="utf-8", newline="") as f:
        f.write(dic)
    with open(pat_in_path, "w", encoding="utf-8", newline="") as f:
        f.write(pat_in)
    return dic_path, pat_in_path


def run_both(dic, pat_in, inputs):
    with tempfile.TemporaryDirectory() as tmp:
        dic_path, pat_in_path = write_manifest(tmp, dic, pat_in)
        out_js = os.path.join(tmp, "out_js.tex")
        out_py = os.path.join(tmp, "out_py.tex")

        js_code, js_stdout, js_stderr = run_js(dic_path, pat_in_path, out_js, inputs)
        assert js_code == 0, (
            f"JS exited with code {js_code}:\n"
            f"STDOUT: {js_stdout}\n"
            f"STDERR: {js_stderr}"
        )

        py_code, py_stdout, py_stderr = run_python(dic_path, pat_in_path, out_py, inputs)
        assert py_code == 0, f"Python exited with code {py_code}:\n{py_stderr}"

        with open(out_js, encoding="utf-8") as f:
            js_content = f.read()
        with open(out_py, encoding="utf-8") as f:
            py_content = f.read()

    return normalize_patterns(js_content), normalize_patterns(py_content), js_stdout, py_stdout


class TestCrossImpl:
    def test_small_single_level_parity(self):
        js_pats, py_pats, js_stdout, _ = run_both(SMALL_DIC, "", SMALL_INPUTS)
        assert len(js_pats) > 0, f"JS produced no patterns:\n{js_stdout}"
        assert js_pats == py_pats, (
            f"Pattern mismatch:\n"
            f"JS  ({len(js_pats)}): {js_pats}\n"
            f"Python ({len(py_pats)}): {py_pats}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Known divergence on multi-level input: JS vs Python produce "
               "different pattern sets (JS 46 vs Python 44 entries; 19/30 words "
               "hyphenate differently). Must be resolved before strict parity.",
    )
    def test_real_multi_level_parity(self):
        """Larger real-word corpus across two hyph levels and pat lengths 1..2.

        KNOWN DIVERGENCE: JS and Python are NOT functionally equivalent here
        (19/30 words hyphenate differently). Frozen as xfail so this is
        tracked; flips to a hard failure once parity is restored.
        """
        js_pats, py_pats, js_stdout, py_stdout = run_both(REAL_DIC, "", MULTI_INPUTS)
        assert len(js_pats) > 0, f"JS produced no patterns:\n{js_stdout}"
        assert len(py_pats) > 0, f"Python produced no patterns:\n{py_stdout}"
        assert js_pats == py_pats, (
            f"Pattern mismatch:\n"
            f"JS  ({len(js_pats)}): {js_pats[:40]}...\n"
            f"Python ({len(py_pats)}): {py_pats[:40]}..."
        )

    def test_golden_output(self):
        """JS output must match the committed golden snapshot (regression net for refactors)."""
        with open(GOLDEN_FILE, encoding="utf-8") as f:
            golden = f.read().splitlines()

        js_pats, _, js_stdout, _ = run_both(REAL_DIC, "", MULTI_INPUTS)
        assert js_pats == golden, (
            f"JS output drifted from golden:\n"
            f"golden ({len(golden)}): {golden[:40]}...\n"
            f"JS ({len(js_pats)}): {js_pats[:40]}..."
        )
        assert len(js_pats) > 0, f"JS produced no patterns:\n{js_stdout}"
