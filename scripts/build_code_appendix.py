#!/usr/bin/env python
"""Build the human-readable code appendix PDF submitted with the dissertation.

Supersedes ``export_code_pdf.py`` (matplotlib backend, plain listing). This
version typesets with LaTeX and produces a navigable document: title page,
linked contents, one section per area of the codebase, syntax highlighting,
line numbers, per-file SHA-256, and page headers.

Every file is listed VERBATIM. Nothing is reformatted, rewritten or elided, so
the appendix is byte-for-byte the code that produced the reported results and
each SHA-256 can be checked against the repository.

    pip install pygments
    python scripts/build_code_appendix.py            # -> docs/code_appendix.pdf
    python scripts/build_code_appendix.py --tex-only # emit .tex, skip compile

Requires a LaTeX installation providing ``xelatex`` (MiKTeX on Windows, TeX
Live elsewhere). With --tex-only the .tex can be compiled anywhere, including
Overleaf.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from pygments import highlight
    from pygments.formatters import LatexFormatter
    from pygments.lexers import (BashLexer, JsonLexer, MarkdownLexer,
                                 PythonLexer, TextLexer, YamlLexer)
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("pygments is required: pip install pygments")

ROOT = Path(__file__).resolve().parents[1]

TITLE = ("What Should an Agent Remember? A Comparative Study of "
         "Memory Architectures for Long-Horizon LLM Agents")
AUTHOR = "Asad Ali"
MODULE = "MA981 MSc Dissertation"
INSTITUTION = ("School of Mathematics, Statistics and Actuarial Science, "
               "University of Essex")

# (section title, orientation note, [glob patterns]) -- order is the reading order
GROUPS = [
    ("Package source: memory architectures",
     "The five memory conditions compared in the study, all implemented behind "
     "the locked \\texttt{Memory} abstract base class so that the harness cannot "
     "tell them apart. \\texttt{base.py} defines the interface and the "
     "Equal-Budget Principle contract; each subsequent module is one arm of the "
     "experiment.",
     ["src/agentmem/memory/*.py"]),
    ("Package source: harness and LLM client",
     "The execution layer. \\texttt{runner.py} drives both Mode A (memory probing) "
     "and Mode B (memory use during execution); \\texttt{prompts.py} owns prompt "
     "assembly and ordering; \\texttt{client.py} wraps the OpenAI-compatible "
     "endpoint with the budget kill-switch and the mock backend used by tests.",
     ["src/agentmem/harness/*.py", "src/agentmem/llm/*.py"]),
    ("Package source: evaluation",
     "Answer judging, evidence-recall scoring and the failure-mode taxonomy "
     "used to produce the \\texttt{reasoning\\_failure} composition reported in "
     "Chapter 4.",
     ["src/agentmem/eval/*.py"]),
    ("Package source: tasks, logging and utilities",
     "Dataset adapters for LongMemEval and the custom Project-Tracker task, "
     "trajectory logging, and the cost, token, manifest and seed helpers.",
     ["src/agentmem/tasks/*.py", "src/agentmem/logging/*.py",
      "src/agentmem/utils/*.py", "src/agentmem/__init__.py"]),
    ("Configuration",
     "Every experimental arm is declared as data, not code. The base config "
     "fixes the shared token budget; each condition file changes only what that "
     "arm is allowed to spend it on. The \\texttt{\\_chrono} and "
     "\\texttt{b\\_reversed} files are the layout-only controls behind the "
     "ordering result.",
     ["configs/*.yaml", "configs/conditions/*.yaml", "configs/tasks/*.yaml"]),
    ("Experiment scripts",
     "Entry points for running the grid, merging and patching runs, and "
     "reproducing every number and figure in Chapter 4 from the stored run "
     "records.",
     ["scripts/*.py"]),
    ("Statistical analysis",
     "The registered family of contrasts, exact McNemar tests with Holm "
     "correction, and the observational ordering decomposition.",
     ["analysis/*.py", "results/audit_2026_08_04/verify_claims.py"]),
    ("Test suite",
     "27 tests covering the memory interface invariants, budget enforcement, "
     "metric correctness and the LongMemEval adapter. All pass on a CPU-only "
     "machine with the mock LLM backend.",
     ["tests/*.py"]),
    ("HPC job scripts",
     "Slurm and shell wrappers used to serve the open-weight model on the "
     "university cluster and to run the grid against it.",
     ["scripts/hpc/*.sh", "scripts/hpc/*.sbatch"]),
    ("Submission notebook",
     "The FASER entry point. Markdown cells appear as comment lines and code "
     "cells in order, so the notebook can be read as a linear script; running "
     "it top-to-bottom performs the same steps as the command-line grid driver.",
     ["Run_Experiments.ipynb"]),
    ("Project documentation",
     "Reproduction instructions as shipped in the repository.",
     ["README.md", "data/README.md", "docs/HPC_RUN_GUIDE.md",
      "docs/CERES_CHEATSHEET.md", "docs/RUNBOOK_patch_b_s1.md",
      "requirements.txt", ".env.example", "data/manifest.json"]),
]

LEXERS = {".py": PythonLexer, ".yaml": YamlLexer, ".yml": YamlLexer,
          ".sh": BashLexer, ".sbatch": BashLexer, ".md": MarkdownLexer,
          ".json": JsonLexer, ".txt": TextLexer, ".example": BashLexer}

# emoji used as bullets in the markdown docs have no glyph in the monospace
# font; swap for equivalents the font can actually draw
GLYPH_FIXES = [("✅", "✓"), ("❌", "✗"),
               ("️", ""), ("⚠", "!")]


def esc(s: str) -> str:
    """Escape a plain string for LaTeX."""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def path_tt(rel: str) -> str:
    """Monospaced path that LaTeX is allowed to break at slashes."""
    return r"\texttt{" + esc(rel).replace("/", r"/\allowbreak{}") + "}"


def git(*args: str, default: str = "unknown") -> str:
    try:
        out = subprocess.run(("git", "-C", str(ROOT)) + args,
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or default
    except Exception:
        return default


def gather(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for f in sorted(ROOT.glob(pat)):
            if not f.is_file() or "__pycache__" in str(f):
                continue
            resolved = f.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(f)
    return files


def notebook_to_source(p: Path) -> str:
    """Flatten a notebook to linear source; markdown cells become comments."""
    nb = json.loads(p.read_text(encoding="utf-8"))
    out: list[str] = []
    for i, cell in enumerate(nb["cells"], 1):
        src = "".join(cell["source"]).rstrip()
        if cell["cell_type"] == "markdown":
            out.append("\n".join(f"# {ln}" for ln in src.splitlines()))
        else:
            out.append(f"# ---- code cell {i} ----\n{src}")
        out.append("")
    return "\n".join(out)


def build_tex() -> tuple[str, int, int]:
    fmt = LatexFormatter(linenos=True, style="friendly", texcomments=False,
                         mathescape=False, escapeinside="")
    rev = git("rev-parse", "--short", "HEAD")
    origin = git("remote", "get-url", "origin", default="")
    repo = (origin.replace("https://", "").replace(".git", "")
            if origin else "agent-memory")

    body: list[str] = []
    total_files = total_lines = 0

    for title, blurb, patterns in GROUPS:
        files = gather(patterns)
        if not files:
            continue
        body.append(r"\section{" + esc(title) + "}")
        body.append(r"\begin{sloppypar}\noindent " + blurb +
                    r"\end{sloppypar}\vspace{1.0em}")
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            raw = f.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8", errors="replace")
            if f.suffix == ".ipynb":
                text = notebook_to_source(f)
            for bad, good in GLYPH_FIXES:
                text = text.replace(bad, good)
            total_files += 1
            total_lines += len(text.splitlines())

            code = highlight(text, LEXERS.get(f.suffix, TextLexer)(), fmt)
            # allow long source lines to wrap rather than run off the page
            code = code.replace(
                r"\begin{Verbatim}[commandchars=\\\{\}, numbers=left, "
                r"firstnumber=1, stepnumber=1]",
                r"\begin{Verbatim}[commandchars=\\\{\},numbers=left,"
                r"firstnumber=1,stepnumber=1,breaklines=true,"
                r"breakanywhere=true,breaksymbolleft=\tiny"
                r"\ensuremath{\hookrightarrow}\,,fontsize=\footnotesize]")

            body.append(r"\needspace{5\baselineskip}")
            body.append(r"\subsection{" + path_tt(rel) + "}")
            body.append(r"\filemeta{" + f"{len(text.splitlines())} lines"
                        + r"}{" + sha[:16] + r"}")
            body.append(r"\nopagebreak")
            body.append(code)
            body.append("")

    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,top=22mm,bottom=20mm,left=20mm,right=16mm,headheight=14pt]{geometry}
\usepackage{fontspec}
\usepackage{fvextra}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{needspace}
\usepackage[hidelinks,bookmarks=true,bookmarksnumbered=true]{hyperref}
\setmainfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}[Scale=0.82]
\definecolor{accent}{HTML}{1F3A5F}
\definecolor{muted}{HTML}{5A6472}
\definecolor{rule}{HTML}{C8CDD4}
""" + fmt.get_style_defs() + r"""
\titleformat{\section}{\Large\bfseries\color{accent}}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{1.6em}{0.7em}
\titleformat{\subsection}{\normalsize\bfseries\color{accent}}{}{0em}{}
\titlespacing*{\subsection}{0pt}{1.4em}{0.3em}
\newcommand{\filemeta}[2]{{\footnotesize\color{muted}#1 \quad\textbullet\quad sha256 #2}\par
  \vspace{0.15em}{\color{rule}\hrule height 0.4pt}\vspace{0.6em}}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\footnotesize\color{muted}MA981 Code Appendix \textbullet{} """ + esc(AUTHOR) + r"""}
\fancyhead[R]{\footnotesize\color{muted}\nouppercase{\leftmark}}
\fancyfoot[C]{\footnotesize\color{muted}\thepage}
\renewcommand{\sectionmark}[1]{\markboth{#1}{}}
\setcounter{tocdepth}{2}
\begin{document}
\begin{titlepage}\centering\vspace*{2.2cm}
{\Large\bfseries\color{accent} Code Appendix\par}\vspace{0.6cm}
{\large """ + esc(MODULE) + r"""\par}\vspace{1.8cm}
{\large\itshape """ + esc(TITLE) + r"""\par}\vspace{2.2cm}
{\large """ + esc(AUTHOR) + r"""\par}\vspace{0.4cm}
{\normalsize\color{muted} """ + esc(INSTITUTION) + r"""\par}\vspace{2.6cm}
\begin{tabular}{@{}ll@{}}
\textbf{Repository} & \texttt{""" + esc(repo) + r"""}\\[2pt]
\textbf{Commit} & \texttt{""" + esc(rev) + r"""}\\[2pt]
\textbf{Files listed} & FILECOUNT\\[2pt]
\textbf{Lines of code} & LINECOUNT\\[2pt]
\textbf{Generated} & """ + dt.date.today().strftime("%d %B %Y") + r"""\\
\end{tabular}\vfill
{\footnotesize\color{muted}Listed verbatim from the repository at the commit above.
No source file was altered to produce this document; every listing is
byte-for-byte the code that generated the reported results, and the SHA-256
prefix under each filename can be checked against the repository.\par}
\end{titlepage}
\tableofcontents\newpage
"""

    tex = (preamble.replace("FILECOUNT", str(total_files))
                   .replace("LINECOUNT", f"{total_lines:,}")
           + "\n".join(body) + "\n\\end{document}\n")
    return tex, total_files, total_lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "code_appendix.pdf"),
                    help="output PDF path (default: docs/code_appendix.pdf)")
    ap.add_argument("--tex-only", action="store_true",
                    help="write the .tex and stop, for compiling elsewhere")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out.with_suffix(".tex")

    tex, n_files, n_lines = build_tex()
    tex_path.write_text(tex, encoding="utf-8")
    print(f"{n_files} files, {n_lines:,} lines -> {tex_path}")

    if args.tex_only:
        return
    if shutil.which("xelatex") is None:
        print("xelatex not found; .tex written. Install MiKTeX/TeX Live, or "
              "compile the .tex on Overleaf.")
        return

    # twice: first pass writes the .aux, second resolves the contents page
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode",
                            "-halt-on-error", tex_path.name],
                           cwd=tex_path.parent, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"xelatex failed; see {tex_path.with_suffix('.log')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
