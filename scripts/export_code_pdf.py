#!/usr/bin/env python
"""Export all project code to ONE machine-readable PDF (FASER requirement).

Uses matplotlib's PDF backend (already a project dependency) so the text is
real selectable text, not images. Scripted early, not submission-eve.

  python scripts/export_code_pdf.py       # -> docs/agentmem_code.pdf
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

INCLUDE = ["src/**/*.py", "scripts/*.py", "tests/*.py", "analysis/*.py",
           "configs/**/*.yaml", "requirements.txt", "README.md",
           "data/README.md", ".env.example"]
LINES_PER_PAGE = 78
CHARS_PER_LINE = 96


def gather() -> list[Path]:
    files: list[Path] = []
    for pat in INCLUDE:
        files.extend(sorted(ROOT.glob(pat)))
    return [f for f in files if f.is_file() and "__pycache__" not in str(f)]


def wrap(line: str) -> list[str]:
    line = line.rstrip("\n").replace("\t", "    ")
    return [line[i:i + CHARS_PER_LINE] for i in range(0, max(len(line), 1), CHARS_PER_LINE)]


def main() -> None:
    out = ROOT / "docs" / "agentmem_code.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = gather()
    with PdfPages(out) as pdf:
        for f in files:
            rel = f.relative_to(ROOT)
            body: list[str] = []
            for n, raw in enumerate(f.read_text(encoding="utf-8",
                                                errors="replace").splitlines(), 1):
                body.extend(f"{n:4d}  {w}" if i == 0 else f"      {w}"
                            for i, w in enumerate(wrap(raw)))
            pages = [body[i:i + LINES_PER_PAGE]
                     for i in range(0, max(len(body), 1), LINES_PER_PAGE)]
            for pno, page in enumerate(pages, 1):
                fig = plt.figure(figsize=(8.27, 11.69))  # A4
                # parse_math=False: source code containing '$...$' (e.g. a
                # docstring mentioning $\iota$) must be typeset literally,
                # not handed to matplotlib's mathtext parser.
                fig.text(0.06, 0.97, f"{rel}  (page {pno}/{len(pages)})",
                         fontsize=9, family="monospace", weight="bold",
                         parse_math=False)
                fig.text(0.06, 0.945, "\n".join(page), fontsize=6.2,
                         family="monospace", va="top", parse_math=False)
                pdf.savefig(fig)
                plt.close(fig)
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({len(files)} files, {size_mb:.1f} MB; FASER limit 50 MB)")


if __name__ == "__main__":
    main()
