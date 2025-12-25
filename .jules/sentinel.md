## 2024-05-22 - [Symlink Traversal Protection]
**Vulnerability:** File overwrite via symbolic links
**Learning:** Python's `os.path.isfile()` returns `True` for symlinks pointing to files, allowing malicious users to trick the application into overwriting sensitive files (like `/etc/passwd` or ssh keys) by supplying a symlink as an output path.
**Prevention:** Explicitly check `os.path.islink(filepath)` before performing file operations, especially writes. This prevents the "Symlink Race" class of vulnerabilities in file handling utilities.
