from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=16,
            firstLineIndent=-8,
            bulletIndent=8,
            spaceAfter=4,
        ),
    }


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def parse_markdown(md_text: str):
    blocks = []
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            blocks.append(("p", " ".join(line.strip() for line in paragraph_lines)))
            paragraph_lines = []

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            blocks.append(("title", stripped[2:].strip()))
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            blocks.append(("h2", stripped[3:].strip()))
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", stripped[2:].strip()))
            continue

        if stripped[:3] in {"1. ", "2. ", "3. ", "4. ", "5. "}:
            flush_paragraph()
            blocks.append(("bullet", stripped))
            continue

        paragraph_lines.append(stripped)

    flush_paragraph()
    return blocks


def render(markdown_path: Path, pdf_path: Path):
    styles = build_styles()
    blocks = parse_markdown(markdown_path.read_text(encoding="utf-8"))

    story = []
    for kind, text in blocks:
        safe = escape(text)
        if kind == "title":
            story.append(Paragraph(safe, styles["title"]))
        elif kind == "h2":
            story.append(Paragraph(safe, styles["h2"]))
        elif kind == "bullet":
            story.append(Paragraph(safe, styles["bullet"], bulletText="•"))
        else:
            story.append(Paragraph(safe, styles["body"]))
        if kind in {"title", "h2"}:
            story.append(Spacer(1, 0.04 * inch))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="PT2 Slides and Final Project Summary",
    )
    doc.build(story)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/render_markdown_pdf.py <input.md> <output.pdf>")

    markdown_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])
    render(markdown_path, pdf_path)


if __name__ == "__main__":
    main()
