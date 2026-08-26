# -*- coding: utf-8 -*-
"""Extrae los outputs de texto del diputraxv13.ipynb ejecutado a un archivo
plano (scratchpad) para redactar la fase 3 (interpretaciones) de plan10upd.md.
No modifica el notebook."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "diputraxv13.ipynb"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scratchpad_v13_outputs.txt"

nb = json.loads(NB.read_text(encoding="utf-8"))
lines = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    texts = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            texts.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                t = "".join(data["text/plain"])
                # los Styler/objetos matplotlib no aportan
                if not t.startswith("<") or "DataFrame" in t:
                    texts.append(t)
        elif o.get("output_type") == "error":
            texts.append("ERROR: " + o.get("ename", "") + ": " + o.get("evalue", ""))
    if texts:
        first = "".join(c["source"]).strip().splitlines()
        head = first[0][:100] if first else ""
        lines.append(f"@@@@@ CELL {i} | {head}")
        lines.append("\n".join(texts).rstrip())
        lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"OK -> {OUT}  ({sum(1 for l in lines if l.startswith('@@@@@'))} celdas con output)")
