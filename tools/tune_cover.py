"""Tunes COVER_BEFORE so each cover line lands where it does on the reference LPU report cover."""
import subprocess, sys, tempfile, json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import report_front
from docx_helpers import Report

KEYS = ["logo", "project", "duration", "title", "sub1", "sub2", "submitted", "name", "reg", "prog", "course", "under", "guide", "desig", "discipline", "school", "univ", "punjab"]
TARGET = dict(logo=71.3, project=211.7, duration=241.2, title=310.3, sub1=344.9, sub2=362.9, submitted=463.7, name=485.3, reg=505.4, prog=523.4, course=542.2,
              under=587.5, guide=609.1, desig=627.8, discipline=648.0, school=666.0, univ=684.7, punjab=703.4)


def render_bands():
    tmp = Path(tempfile.mkdtemp()); rp = Report(); report_front.cover(rp); d = tmp / "cover.docx"; rp.save(d)
    subprocess.run([sys.executable, "/mnt/skills/public/docx/scripts/office/soffice.py", "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(d)], capture_output=True, timeout=300)
    subprocess.run(["pdftoppm", "-png", "-r", "100", "-f", "1", "-l", "1", str(tmp / "cover.pdf"), str(tmp / "c")], capture_output=True)
    img = np.array(Image.open(next(tmp.glob("c-*.png"))).convert("RGB")); dark = img.sum(axis=2) < 600
    rows = dark.any(axis=1); bands, s = [], None
    for y, v in enumerate(rows):
        if v and s is None: s = y
        if not v and s is not None: bands.append((s, y - 1)); s = None
    pages = subprocess.run(["pdfinfo", str(tmp / "cover.pdf")], capture_output=True, text=True).stdout
    return [(b[0] * 0.72, b[1] * 0.72) for b in bands], int([l for l in pages.splitlines() if l.startswith("Pages")][0].split()[1])


for it in range(5):
    bands, pages = render_bands()
    print(f"iteration {it}: bands={len(bands)} pages={pages}")
    if len(bands) != len(KEYS): print("  band count mismatch:", [round(b[0], 1) for b in bands]); break
    delta = {k: TARGET[k] - bands[i][0] for i, k in enumerate(KEYS)}
    print("  max |error| pt:", round(max(abs(v) for v in delta.values()), 1))
    if max(abs(v) for v in delta.values()) < 1.0: break
    prev = 0.0
    for k in KEYS:
        if k != "sub2": report_front.COVER_BEFORE[k] = round(max(0.0, report_front.COVER_BEFORE[k] + (delta[k] - prev)), 1)
        prev = delta[k]
print(json.dumps(report_front.COVER_BEFORE))
