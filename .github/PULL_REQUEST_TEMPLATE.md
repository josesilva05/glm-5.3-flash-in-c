## What this changes

<!-- One or two sentences. What behaviour is different after this PR? -->

## Why

<!-- The problem being solved. Link an issue if there is one. -->

## Verification

- [ ] `ctest --test-dir build -C Release` (or `make test`) passes: tiny oracle, config refusals, safetensors
- [ ] If a kernel changed: `tools/make_glm_tiny.py` still produces a fixture the engine matches, and a
      deliberate mutation of the changed kernel makes `test_glm_tiny` fail
- [ ] If the tokenizer changed: `tools/tok_parity_glm.py` passes against the real tokenizer
- [ ] If output could change on the real model: `tools/glm_reference.py` logits match `glm53f --dump-logits`

## Numbers, if this is a performance change

<!-- Report at least 3 runs of each arm; disk-bound timings vary a lot between runs. -->

| arm | run 1 | run 2 | run 3 | mean |
|-----|-------|-------|-------|------|
|     |       |       |       |      |

## Risk

<!-- What could this break that the tests would not catch? -->
