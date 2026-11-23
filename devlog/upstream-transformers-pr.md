# Transformers PR: import torchaudio lazily in `audio_utils`

Opened as [huggingface/transformers#51099](https://github.com/huggingface/transformers/pull/51099)
on 2026-11-22, from branch `fix-lazy-torchaudio-import` of `devt-code/transformers` (commit
`533b01c`, on top of `main` at `be48e7a`), with a regression test. **Merged on 2026-11-23**
(merge commit `d1fc2bc`), about 17 hours after it was opened.

## How the review went

- A bot held back the full CI: it runs on its own only for authors who already have a merged PR
  in the repo, so a maintainer had to approve it.
- About 12 hours in, Rocketknight1 triaged it and passed it to the audio maintainers (eustlb,
  ebezzam).
- eustlb pushed a small "nits" commit to the branch (a comment that says why torchaudio loads
  lazily instead of pointing at #47422, and `@require_torch` on the new test), approved the PR and
  turned on auto-merge.
- CI passed and it merged on its own. The earlier plan, tagging vasqu and ydshieh once CI was
  green, wasn't needed.

## The bug, in plain words

Transformers loads `audio_utils.py` whenever any model is loaded, and that file loaded torchaudio at
the top. When torchaudio is installed but broken, as on our Colab run (built for CUDA 13.0 next to
a PyTorch built for CUDA 13.2), even a text-only model like Qwen3 fails to load, with an error that
hides the cause. The fix loads torchaudio inside the one function that uses it. It's the same fix
the maintainers applied to `loss/loss_rnnt.py` in July (#47422), after a broken torchaudio took down
their daily CI.

## Checks done

| Check | Result |
|---|---|
| Reproducer below | Fails on `main` (exit 1, the Colab error); passes on the branch (`loaded Qwen3ForCausalLM`) |
| New regression test | Fails on `main` with the `RuntimeError`, passes with the fix |
| `tests/utils/test_import_utils.py` | 173 passed, 1 skipped |
| `make fix-repo` | All 17 checks pass, no files changed |
| `ruff check`, `ruff format --check` (0.14.10) | Pass |
| Audio utils, import structure, Whisper and wav2vec2 processor tests | Same results as `main`; the local failures come from optional audio packages missing on this Mac |
| Broken torchaudio, `load_audio(path, backend="torchaudio")` | Raises torchaudio's own `RuntimeError` at the call site; nothing is masked |
| Healthy torchaudio, same call | Calls `torchaudio.load` and resamples as before |

The rules this PR follows (`CONTRIBUTING.md`, "Contribution rules"): a standalone PR is allowed for
a bug nobody has reported, if it includes a script that fails on `main` and passes with the fix. No
need to ask first; the first PR for a problem gets priority. Wait for CI to pass before tagging
reviewers.

## How to open it

1. Fork https://github.com/huggingface/transformers with the Fork button.
2. Push the branch to your fork:
   ```bash
   cd ~/Documents/Projects/transformers
   git remote add fork https://github.com/devt-code/transformers.git
   git push fork fix-lazy-torchaudio-import
   ```
3. Open the pull request from the link `git push` prints, using the title and description below.

## Title

Import torchaudio lazily in `audio_utils` so a broken torchaudio doesn't break model imports

## Description

````markdown
# What does this PR do?

`audio_utils.py` imports torchaudio at module scope. Every model reaches it through `modeling_layers` → `processing_utils` → `audio_utils`, so a torchaudio that fails to load makes even text-only models fail to import, with an error that hides the cause:

```
ModuleNotFoundError: Could not import module 'Qwen3ForCausalLM'. Are this object's requirements defined correctly? Set the logging verbosity to DEBUG for the original import error.
```

I hit this on Google Colab: after `pip install torch==2.15.0`, the preinstalled torchaudio 2.11.0+cu130 fails on import with `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 13.2 whereas TorchAudio has CUDA version 13.0.` The newest torchaudio on PyPI is 2.11.1, older than current torch releases.

This applies the fix from #47422 (`loss/loss_rnnt.py`) to `audio_utils.py`: torchaudio is imported inside the torchaudio branch of `load_audio`, the only code that uses it. Importing a model no longer touches torchaudio, a broken install still raises its own error when `backend="torchaudio"` is used, and a missing one still gets the `requires_backends` error.

## Reproducer

Fails on `main` (be48e7a) with the error above and passes with this PR, printing `loaded Qwen3ForCausalLM`. It stands in for a broken torchaudio with a package that raises on import.

```python
import os
import pathlib
import subprocess
import sys
import tempfile

broken = pathlib.Path(tempfile.mkdtemp())
(broken / "torchaudio").mkdir()
(broken / "torchaudio" / "__init__.py").write_text(
    'raise RuntimeError("Detected that PyTorch and TorchAudio were compiled with different CUDA versions.")\n'
)
(broken / "torchaudio-2.11.0.dist-info").mkdir()
(broken / "torchaudio-2.11.0.dist-info" / "METADATA").write_text(
    "Metadata-Version: 2.1\nName: torchaudio\nVersion: 2.11.0\n"
)

code = """
import torch
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
print("loaded", type(model).__name__)
"""
result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(broken)})
sys.exit(result.returncode)
```

## Tests

- New `test_broken_torchaudio_does_not_break_model_import`, next to `test_broken_torchaudio_does_not_break_import`: it makes `import torchaudio` raise, reports torchaudio as installed, and re-imports `audio_utils` and `modeling_llama`. It fails on `main` with the `RuntimeError` and passes with this change.
- `tests/utils/test_import_utils.py`: 173 passed, 1 skipped.
- `make fix-repo`: all checks pass, no changes.
- `tests/utils/test_audio_utils.py`, `tests/utils/test_import_structure.py`, and the Whisper and wav2vec2 processor tests: same results before and after.

## Who can review?

@vasqu @ydshieh, who reviewed #47422
````
