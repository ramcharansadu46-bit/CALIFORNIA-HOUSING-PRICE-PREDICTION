"""Two-pass assembly of the DOCX report:  python tools/build_report.py
Pass 1 builds with placeholder page numbers; the file is rendered to PDF and heading/caption pages are located;
pass 2 rebuilds with real page numbers cached in the TOC / list-of-figures / list-of-tables fields."""
import json, re, subprocess, sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from docx_helpers import Report
import rc, report_front, report_s1_3, report_s4a, report_s4b, report_end
from rc import ROOT

OUT = ROOT / "report" / "CA2_Predictive_Analysis_Report.docx"


def build(page_map=None):
    R = rc.load(); rp = Report(page_map)
    report_front.cover(rp)
    rp.new_section(fmt="lowerRoman", start=1, footer=True)           # front matter, roman numerals
    report_front.declaration(rp); rp.page_break()
    report_front.certificate(rp); rp.page_break()
    report_front.acknowledgement(rp); rp.page_break()
    report_front.front_heading(rp, "TABLE OF CONTENTS"); toc_anchor = rp.p("")
    rp.page_break()
    report_front.front_heading(rp, "LIST OF FIGURES"); lof_anchor = rp.p("")
    rp.page_break()
    report_front.front_heading(rp, "LIST OF TABLES"); lot_anchor = rp.p("")
    rp.new_section(fmt="decimal", start=1, footer=True)               # body, arabic numerals from 1
    report_s1_3.introduction(rp, R); report_s1_3.source(rp, R); report_s1_3.eda(rp, R)
    rp.h(1, "4. Analysis on Dataset")
    rp.p("This section presents the six analyses that answer the research objectives of Section 1.3. Each analysis follows the same structure: an introduction, a general description, the specific requirements, functions and formulas, the actual analysis results, and the visualisation with its interpretation. All numerical results were produced by the executed notebook and are stored in *results/metrics.json*.")
    report_s4a.o1(rp, R); report_s4a.o2(rp, R); report_s4a.o3(rp, R); report_s4a.o4(rp, R)
    report_s4b.o5(rp, R); report_s4b.o6(rp, R)
    report_s4b.conclusion(rp, R)
    report_end.future_scope(rp, R); report_end.references(rp, R)
    pm = page_map or {}
    heads = [(l, t.replace("*", ""), pm.get(t, "0")) for l, t in rp.headings]
    rp.toc_block(toc_anchor, 'TOC \\o "1-3" \\h \\z \\u', heads, lambda l: f"TOC {l}")
    rp.toc_block(lof_anchor, 'TOC \\h \\z \\c "Figure"', [(1, c, pm.get(c, "0")) for c in rp.figures], lambda l: "Table of Figures")
    rp.toc_block(lot_anchor, 'TOC \\h \\z \\c "Table"', [(1, c, pm.get(c, "0")) for c in rp.tables], lambda l: "Table of Figures")
    return rp


def _soffice_cmd():
    """LibreOffice launcher: works on any machine that has `soffice` on PATH."""
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if exe: return [exe]
    sandbox = "/mnt/skills/public/docx/scripts/office/soffice.py"       # only exists in the build environment
    if Path(sandbox).exists(): return [sys.executable, sandbox]
    raise SystemExit("LibreOffice ('soffice') must be installed and on PATH to compute page numbers for the contents.")


def to_pdf(docx_path, outdir):
    subprocess.run(_soffice_cmd() + ["--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx_path)], capture_output=True, timeout=600)
    return Path(outdir) / (Path(docx_path).stem + ".pdf")


def page_texts(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout.split("\f")
    return n, [t for t in txt][:n]


def norm(s): return re.sub(r"\s+", " ", s.replace("*", "")).strip()


def locate(rp, pages):
    """Return {heading/caption text: printed page number} using the PDF text."""
    # first body page = first page containing a line that is exactly the first H1
    first_h1 = norm(rp.headings[0][1])
    body0 = next(i for i, t in enumerate(pages) if any(norm(l) == first_h1 for l in t.split("\n")))
    pm, cursor = {}, body0
    for (lvl, text) in rp.headings:
        key = norm(re.sub(r"_\{[^}]*\}|\^\{[^}]*\}", "", text))
        for i in range(cursor, len(pages)):
            if any(norm(l).startswith(key[:60]) and "...." not in l for l in pages[i].split("\n")):
                pm[text] = str(i - body0 + 1); cursor = i; break
    for caps in (rp.figures, rp.tables):
        cur = body0
        for c in caps:
            key = norm(c)[:45]
            for i in range(cur, len(pages)):
                if any(norm(l).startswith(key) for l in pages[i].split("\n")):
                    pm[c] = str(i - body0 + 1); cur = i; break
    return pm, body0


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); tmp = ROOT / "report" / "_tmp"; tmp.mkdir(exist_ok=True)
    rp = build(); p1 = tmp / "pass1.docx"; rp.save(p1)
    pdf = to_pdf(p1, tmp); n, pages = page_texts(pdf); pm, body0 = locate(rp, pages)
    rp2 = build(pm); rp2.save(OUT)
    pdf2 = to_pdf(OUT, tmp); n2, pages2 = page_texts(pdf2)
    missing = [t for _, t in rp2.headings if t not in pm] + [c for c in rp2.figures + rp2.tables if c not in pm]
    print(f"pages pass1={n} pass2={n2}; front-matter pages={body0}; body pages={n2 - body0}; headings={len(rp2.headings)} figures={len(rp2.figures)} tables={len(rp2.tables)}; unlocated={len(missing)}")
    for m in missing[:10]: print("  unlocated:", m)
    shutil.copy(pdf2, ROOT / "report" / "CA2_Predictive_Analysis_Report_preview.pdf")
