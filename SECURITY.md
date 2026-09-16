# Security

## Reporting

Report vulnerabilities privately through GitHub's "Report a vulnerability" flow rather
than a public issue.

## Scope

**The trust boundary is the model files.** This engine parses binary and JSON input that
users routinely download from third-party mirrors: safetensors headers, config.json and
the tokenizer files. It treats all of them as untrusted.

The parsers bound what they read (nesting depth, header length, digit strings, tensor
shapes and data offsets) and refuse implausible values rather than trusting the file. A
crafted checkpoint that causes an out-of-bounds read or write, an unbounded allocation,
or execution of attacker-controlled data is a vulnerability, and is in scope.

The model WEIGHTS themselves are not a trust boundary this engine can defend: a
checkpoint with valid structure and hostile parameter values will produce hostile output,
and no parser check can prevent that. Verify downloaded shards against the published
hashes if your threat model includes a hostile mirror.

Out of scope: the quality or safety of generated text, and denial of service through
legitimately large models.

## Credentials

Nothing in this repository reads credentials. `.gitignore` excludes `hf_token*` and key
material; keep tokens out of logs if you add tooling that needs them.
