# Upstream contribution plan

Updated 2026-11-23. A Transformers fix was merged on 2026-11-23 as
[huggingface/transformers#51099](https://github.com/huggingface/transformers/pull/51099): a bug
this project ran into on Colab, which nobody had reported. Every issue-tracker candidate checked
earlier already had someone else's PR.

## Merged: Transformers

A broken torchaudio made every model import fail, even text-only ones, because `audio_utils.py`
imported torchaudio at module scope. The fix imports it lazily, the same way the maintainers fixed
`loss_rnnt.py` in July. Branch `fix-lazy-torchaudio-import` on `devt-code/transformers`; local
clone in `~/Documents/Projects/transformers`. An audio maintainer pushed a small cleanup commit,
approved it, and it merged about 17 hours after it was opened. Checks, the PR text and how the
review went: [upstream-transformers-pr.md](upstream-transformers-pr.md).

Transformers accepts a standalone PR for a bug nobody has reported, if it includes a script that
fails on `main` and passes with the fix. No need to ask first; the first PR gets priority.

## Earlier candidates, all taken (checked 2026-11-22)

| Issue | Status |
|---|---|
| vLLM [#33267](https://github.com/vllm-project/vllm/issues/33267), [#32335](https://github.com/vllm-project/vllm/issues/32335), [#27433](https://github.com/vllm-project/vllm/issues/27433) | Assigned to others |
| vLLM [#40004](https://github.com/vllm-project/vllm/issues/40004), [#66698](https://github.com/vllm-project/vllm/issues/66698), [#56653](https://github.com/vllm-project/vllm/issues/56653), [#69139](https://github.com/vllm-project/vllm/issues/69139) | Open PRs from others, 1–4 each |
| vLLM: fork fails after CUDA starts (hit on Colab) | Open PRs #68902 and #67987 |
| guidellm [#1565](https://github.com/vllm-project/guidellm/issues/1565), [#1522](https://github.com/vllm-project/guidellm/issues/1522), [#1529](https://github.com/vllm-project/guidellm/issues/1529), [#1509](https://github.com/vllm-project/guidellm/issues/1509), [#1531](https://github.com/vllm-project/guidellm/issues/1531), [#1588](https://github.com/vllm-project/guidellm/issues/1588), [#1616](https://github.com/vllm-project/guidellm/issues/1616) | Open PRs from others |
| guidellm [#1592](https://github.com/vllm-project/guidellm/issues/1592) | Maintainer: not open to outside contributions |
| guidellm [#1456](https://github.com/vllm-project/guidellm/issues/1456) | Covered by PR #1339 |
| nano-vllm [#279](https://github.com/GeeeekExplorer/nano-vllm/issues/279), [#274](https://github.com/GeeeekExplorer/nano-vllm/issues/274), [#240](https://github.com/GeeeekExplorer/nano-vllm/issues/240) | Open PRs from others; no merges since April 2026 |

Lesson: beginner-friendly issues in popular repos now draw PRs within hours. Bugs you hit yourself,
with a reproducer, are the reliable way in.

## How to do it next time
1. Search the repo's issues and PRs for the problem first; if someone reported or fixed it, don't
   open another PR.
2. Reproduce it with a failing test before changing code.
3. Keep the diff minimal, reuse an approach the codebase already has, and link to it.
4. Run the repo's own checks (`make fix-repo` in Transformers) and the tests for the files you
   changed, and wait for CI to pass before tagging reviewers.
5. On a first PR, expect the full CI to wait for a maintainer's approval. In Transformers a triage
   maintainer routes new PRs to the right reviewers, so tagging people yourself may not be needed.
