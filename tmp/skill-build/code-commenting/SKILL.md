---
name: code-commenting
description: Add sparse, high-signal comments to source code in a comments-only pass. Use when Codex is asked to add, improve, rewrite, or remove code comments; explain tricky logic inline; document invariants or cross-module contracts; fix stale comments that no longer match the code; or clean up empty section headers and unnecessary blank spacing around comment blocks. Prefer this skill when comments should clarify intent and non-obvious decisions rather than narrate syntax, when executable behavior must remain unchanged, and when vendor or API-origin code should be left untouched unless explicitly requested.
---

# Code Commenting

Review the code before writing comments. Add as little commentary as needed to make the non-obvious parts understandable to a competent engineer.

## Safety Contract

Treat a commenting pass as comments-only by default.

Do not change:

- executable logic
- names, signatures, or imports
- control flow
- data structures
- configuration values
- formatting beyond the minimum spacing cleanup needed to keep comment headers attached to code

If a real code fix seems necessary, report it instead of making it during the commenting pass.

## Ownership Boundary

Comment only project-authored code by default.

Do not edit comments in:

- Delsys-provided API code
- vendor or upstream reference code
- code that appears copied from external SDK examples

If a file mixes project code with vendor-origin code, limit the pass to the project-authored additions. If ownership is unclear, leave that block untouched and note the ambiguity.

## Comment Standard

Add comments for:

- intent the code does not reveal by itself
- invariants, assumptions, and contracts that must stay true
- surprising implementation choices or workarounds
- behavior that spans multiple functions, files, or runtime phases
- edge-case handling that would otherwise look arbitrary

Do not add comments for:

- obvious assignments, arithmetic, or control flow
- line-by-line restatement of what the code already says
- comments that duplicate good names or clear function signatures
- historical notes unless they still affect present behavior
- noisy banner comments or documentation blocks with no real information

Delete or rewrite comments that are:

- stale because the code, data, config, or active users changed
- placeholder section headers with nothing under them
- misleading because they describe removed behavior
- separated from the code by unnecessary blank lines

## Workflow

1. Read the surrounding code first.
2. Identify which files or blocks are project-authored versus vendor-origin.
3. Remove or rewrite stale comments before adding new ones.
4. Identify only the blocks where a reader would likely ask "why is this here?" or "what assumption is this relying on?"
5. Prefer one short comment above a block over many inline comments.
6. Keep wording concrete and factual.
7. Match the file's existing tone and comment style.
8. Collapse unnecessary blank lines around comment headers and code blocks.
9. Run the lightest available verification that shows the file still parses, builds, or runs after the comment-only edit.
10. Re-read after editing and remove any comment that feels redundant.

## Style Rules

- Prefer short comments over paragraphs.
- Explain why before what.
- Use plain technical language; avoid filler.
- Keep comments durable so they stay true after small refactors.
- Prefer comments attached to a logical block, not every statement inside it.
- If a section header comment remains, keep it directly attached to the code it introduces.
- Report the verification you ran, or state clearly when no verification was available.
- Leave Delsys/vendor-origin comments untouched unless the user explicitly asks to modify them.

## Examples

Good:

```python
# Clamp to the overlap window so every channel is interpolated onto the same
# time base before training windows are extracted.
t_grid = build_common_grid(timestamps, target_fs_hz)
```

Bad:

```python
t_grid = build_common_grid(timestamps, target_fs_hz)  # build the time grid
```

Bad:

```python
# -- Calibration -------------------------------------------------------------



# -- Data loading ------------------------------------------------------------

def load_dataset():
```

Good:

```python
# -- Data loading ------------------------------------------------------------
def load_dataset():
```
