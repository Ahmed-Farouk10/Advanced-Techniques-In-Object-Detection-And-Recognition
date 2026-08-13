# Research Paper — Editing Manual

> **Branch:** `research-paper` | **Paper:** *Smoke Before Fire: Can Object Detectors Find a Wildfire Before the Flame Is Visible?*
> **Authors:** Esraa Nasr ElSayed, Ahmed Ayman

This branch contains the complete research paper, written in LaTeX, plus all supporting documentation and the compiled PDF.

---

## Files You Need to Know

| File | What It Is | Edit When |
|------|-----------|-----------|
| `paper/cognitive_fire_defense.tex` | **The paper itself** (LaTeX source) | Any change to content, wording, tables, citations |
| `paper/cognitive_fire_defense_final.pdf` | The compiled PDF | Regenerate after editing `.tex` |
| `paper/compile_pdf.js` | Puppeteer PDF compiler | Only if PDF rendering breaks |
| `paper/PAPER.md` | Metadata (title, authors, abstract, keywords) | Author list or high-level info changes |
| `paper/logic/` | Research scaffolding (claims, experiments, concepts, problem) | Only when the *science* changes, not the wording |
| `paper/literature_comparison_matrix.md` | 7 comparison tables vs 12 papers | When adding/updating citations |
| `paper/references/` | 12 downloaded reference PDFs | **Not tracked in git** (copyrighted) |
| `Rules.md` | Q1 journal compliance checklist | When target venue rules change |

---

## How to Edit the Paper

### Step 1 — Open the source

Open `paper/cognitive_fire_defense.tex` in **VS Code** (install the "LaTeX Workshop" extension) or any text editor.

The file is plain text with LaTeX markup. The content between `\section{...}` commands is your prose. Math is wrapped in `$...$` (inline) or `\begin{equation}...\end{equation}` (displayed).

### Step 2 — Find the section

| Section | Opens with | Line location (approx) |
|---------|-----------|------------------------|
| Title + authors | `\title{...}` + `\author{...}` | Top of file |
| Abstract | `\begin{abstract}` | ~line 17 |
| Introduction | `\section{Introduction}` | ~line 32 |
| Related Work | `\section{Related Work}` | ~line 90 |
| Materials & Methods | `\section{Materials and Methods}` | ~line 140 |
| Experimental Setup | `\section{Experimental Setup}` | ~line 380 |
| Results | `\section{Results}` | ~line 430 |
| Discussion | `\section{Discussion}` | ~line 435 |
| Conclusion | `\section{Conclusion}` | ~line 445 |
| References | `\begin{thebibliography}` | ~line 460 |

### Step 3 — Make your edit

- **Fix a sentence:** find it, type over it.
- **Add a citation:** type `~\cite{key}` where `key` matches a `\bibitem{key}` entry in the bibliography at the bottom.
- **Add a table row:** follow the existing row pattern inside the `\begin{tabular}` block. Columns separated by `&`, rows ended with `\\`.
- **Add a figure:** `\includegraphics[width=\columnwidth]{figures/yourfile.png}` — put the image in `paper/figures/`.
- **Add emphasis:** `\textbf{bold}`, `\textit{italic}`.
- **Subscript/superscript:** `AP\textsubscript{small}`, `64\textsuperscript{2}`.

### Step 4 — Compile to PDF

**Option A (recommended, no LaTeX needed):**

```powershell
cd paper
node compile_pdf.js
```

This renders `cognitive_fire_defense_final.pdf` via Puppeteer. Requires `npm install puppeteer` first time.

**Option B (if you have a working LaTeX install):**

```powershell
cd paper
pdflatex cognitive_fire_defense.tex
pdflatex cognitive_fire_defense.tex
```

Run twice to resolve cross-references.

---

## Common Edits Cheat Sheet

| You Want To | Do This |
|-------------|---------|
| Change a number in the abstract | Edit the sentence in the abstract block |
| Fix a table value | Find the number inside `\begin{tabular}` and edit it |
| Add a new reference | Add `\bibitem{newkey}` at the bottom, then `\cite{newkey}` in the text |
| Remove a sentence | Delete it from the prose (not from any `\` command) |
| Change author order | Edit `\author{...}` near the top |
| Rename a section | Change the text inside `\section{...}` |

---

## What NOT to Edit Manually

- `paper/logic/*.md` — these are the research design scaffold. Changing claims/experiments here without updating the `.tex` will desync the paper. Tell Ahmed if the science changes.
- `paper/literature_comparison_matrix.md` — regenerated from the survey. If you add a paper, update both this and the References section in `.tex`.

---

## Git Workflow

```powershell
# See what changed
git status

# Commit your edits
git add paper/cognitive_fire_defense.tex
git commit -m "Your message describing the edit"

# Push to GitHub
git push origin research-paper
```

**Do NOT commit:**
- `paper/references/` (copyrighted PDFs)
- `paper/node_modules/`
- Dataset images (already gitignored)
- Generated HTML preview files

---

## Gotchas (Common LaTeX Mistakes)

1. **Missing `$` around math** — `k=5` inside prose must be `$k=5$` or it breaks.
2. **Unescaped `%`** — a literal percent sign in text must be `\%`, otherwise LaTeX treats it as a comment.
3. **Unescaped `&`** — in prose, an ampersand must be `\&`. Inside tables, `&` is the column separator (leave it).
4. **Mismatched braces** — every `{` needs a `}`.
5. **Special characters** — `#`, `_`, `$`, `%`, `&`, `~`, `^`, `\` all need escaping in prose.

If you see a cryptic compile error, check for these five things first.

---

## Questions?

Ask Ahmed before editing anything structural (contributions list, methodology order, claim wording). Wording tweaks and number fixes are safe to do directly.
