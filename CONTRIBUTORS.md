# Contributors

People whose fixes or features are part of the code in this repository.

- **[genesisrevelationinc-debug](https://github.com/genesisrevelationinc-debug)** --
  native Windows support (`src/io/glm53f_portable_io.h`), including the heap-corruption
  bug from pairing `_aligned_malloc` with plain `free()`.
- **[ysgao](https://github.com/ysgao)** -- the Darwin `pread()` 2 GiB per-call ceiling
  that large contiguous reads hit, handled by `GLM53F_PREAD_MAX`.
- **[douglasmun](https://github.com/douglasmun)** -- macOS / Apple Silicon build support.
- **[ShaalanMarwan](https://github.com/ShaalanMarwan)** -- applying the x86-only
  `-mavx2`/`-mfma` flags only on x86 in the CMake build, so ARM64 builds work.
- **[biokraft](https://github.com/biokraft)** -- the CI job that builds and tests the
  CMake path rather than only documenting it as equivalent to Make.

Thank you.
