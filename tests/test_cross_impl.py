"""Cross-implementation test: verify JS and Python produce identical patterns from same input."""

import os
import sys
import tempfile
import subprocess
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DIC_CONTENT = "ka-ta\nba-cu\nsu-su\n"
PAT_IN_CONTENT = ""
INPUTS = "1 1\n1 1\n1 1\n1 1 1\nn\nn"


def run_js(dic_path, pat_in_path, pat_out_path):
    import time
    proc = subprocess.Popen(
        ["node", "orthos.js", dic_path, pat_in_path, pat_out_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=os.path.join(os.path.dirname(__file__), "..")
    )
    for line in INPUTS.split("\n"):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        time.sleep(0.4)
    proc.stdin.close()
    stdout, stderr = proc.communicate(timeout=30)
    return proc.returncode, stdout, stderr


def run_python(dic_path, pat_in_path, pat_out_path):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "orthos_colab.py", dic_path, pat_in_path, pat_out_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", cwd=os.path.join(os.path.dirname(__file__), ".."), env=env
    )
    stdout, stderr = proc.communicate(input=INPUTS, timeout=60)
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


class TestCrossImpl:
    def test_js_and_python_produce_identical_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dic_path = os.path.join(tmp, "test.dic")
            pat_in_path = os.path.join(tmp, "in.pat")
            pat_out_js = os.path.join(tmp, "out_js.tex")
            pat_out_py = os.path.join(tmp, "out_py.tex")

            with open(dic_path, "w", encoding="utf-8", newline="") as f:
                f.write(DIC_CONTENT)
            with open(pat_in_path, "w", encoding="utf-8", newline="") as f:
                f.write(PAT_IN_CONTENT)

            js_code, js_stdout, js_stderr = run_js(dic_path, pat_in_path, pat_out_js)
            assert js_code == 0, (
                f"JS exited with code {js_code}:\n"
                f"STDOUT: {js_stdout}\n"
                f"STDERR: {js_stderr}"
            )

            py_code, py_stdout, py_stderr = run_python(dic_path, pat_in_path, pat_out_py)
            assert py_code == 0, f"Python exited with code {py_code}:\n{py_stderr}"

            with open(pat_out_js, encoding="utf-8") as f:
                js_content = f.read()
            with open(pat_out_py, encoding="utf-8") as f:
                py_content = f.read()

            js_pats = normalize_patterns(js_content)
            py_pats = normalize_patterns(py_content)

            assert len(js_pats) > 0, f"JS produced no patterns:\n{js_stdout}"
            assert len(py_pats) > 0, f"Python produced no patterns:\n{py_stdout}"

            assert js_pats == py_pats, (
                f"Pattern mismatch:\n"
                f"JS  ({len(js_pats)}): {js_pats}\n"
                f"Python ({len(py_pats)}): {py_pats}"
            )
