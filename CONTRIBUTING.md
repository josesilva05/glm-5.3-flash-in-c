# Contributing

Fixes and features from outside the project are credited in
[CONTRIBUTORS.md](CONTRIBUTORS.md).

## Before anything

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

(`make -j && make test` on Linux/macOS/MinGW.) The tests need no model weights and must
stay green.

The Python tools need `torch`, `transformers`, `safetensors` and `numpy` (versions in
`pyproject.toml`):

- `tools/glm_reference.py` runs the official `modeling_glm5_next.py` modules layer by
  layer on the real checkpoint and writes the reference logits;
- `tools/make_glm_tiny.py` regenerates the weightless oracle in `tests/fixtures/glm_tiny`;
- `tools/tok_parity_glm.py` checks the C tokenizer against the official one.

## The standard this codebase holds itself to

**A wrong answer that looks right is the worst failure mode here.** The engine can load a
mis-bound tensor, stream a corrupt expert or mis-tokenize a prompt and still emit fluent,
plausible text. Nothing crashes.

- **Fail loudly, never silently.** A missing config field or tensor fails the load; a
  failed expert read fails the run. Never substitute a default, NULL or zeros.
- **A test that cannot fail is not a test.** When you change a kernel, break it on
  purpose and confirm `test_glm_tiny` fails; if it does not, make the tiny model exercise
  that path (as its low `swiglu_limit` does for the clamp).
- **The reference is the upstream implementation.** Name the module a kernel mirrors at
  its definition, and validate real-model changes with `tools/glm_reference.py`.

## Performance changes

Disk-bound timings vary a lot between runs. Report at least three runs per arm, and
prefer counts (expert bytes read, cache retention) over seconds where possible.

## Style

- C99, 4-space indent, 90 columns (`.clang-format`).
- Keep `-ffp-contract=off` (`/fp:precise` on MSVC).
- `-Wpointer-arith` is deliberate: weight pointers are `const void *`, and arithmetic on
  void strides by one byte under GCC.

## Commits and PRs

Present tense, imperative. Explain *why* in the body; fill in the PR template's
verification section.
