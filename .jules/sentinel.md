# Security Journal

## Testing CLI User Inputs safely
When testing interactive CLI input helpers that loop indefinitely (`while True`), ensure mock input iterators yield valid input at the end to avoid `StopIteration` unhandled exceptions in tests.
When verifying error messages written to `stdout`, `pytest`'s `capsys` fixture allows inspecting standard output without affecting execution.
