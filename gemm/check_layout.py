# Programmatic layout gate: re-executes gemm.py in-process, then checks every
# Text artist in every axes for (a) pairwise text-on-text overlaps and
# (b) text extending beyond its axes bbox. Prints any violations with locations.
# All extents are measured through one renderer after a single draw() so axes
# and text boxes share the same dpi; bboxes are frozen to avoid lazy re-eval.

import runpy
import matplotlib.text as mtext
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

g = runpy.run_path(r"C:\Users\barry\Desktop\ml-infra-journey\physics-llm-inference-working\gemm\gemm.py")
fig = g["fig"]
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

def frozen(bb):
    return Bbox([[bb.x0, bb.y0], [bb.x1, bb.y1]])

panel_names = ["A anatomy", "B intensity-vs-n", "C roofline",
               "D prefill-vs-decode", "E benchmark", "F takeaway"]
issues = 0

for idx, ax in enumerate(fig.axes):
    name = panel_names[idx] if idx < len(panel_names) else f"axes {idx}"
    ax_bbox = frozen(ax.get_window_extent(renderer))
    texts = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        # base-class call: Annotation.get_window_extent unions in the arrow
        try:
            bb = frozen(mtext.Text.get_window_extent(t, renderer))
        except Exception:
            continue
        texts.append((t.get_text().replace("\n", " / ")[:46], bb))
    # tick labels too (they live on the axis, not in ax.texts)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        if tl.get_text().strip():
            texts.append((f"[tick] {tl.get_text()}", frozen(tl.get_window_extent(renderer))))
    # (a) pairwise overlaps, ignore slivers under 3 px deep in either dimension
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            x0 = max(texts[i][1].x0, texts[j][1].x0)
            x1 = min(texts[i][1].x1, texts[j][1].x1)
            y0 = max(texts[i][1].y0, texts[j][1].y0)
            y1 = min(texts[i][1].y1, texts[j][1].y1)
            if x1 - x0 > 3 and y1 - y0 > 3:
                issues += 1
                print(f"OVERLAP  [{name}] '{texts[i][0]}'  <->  '{texts[j][0]}'  "
                      f"({x1-x0:.0f}x{y1-y0:.0f} px)")
    # (b) clipping beyond axes bbox (6 px tolerance; titles above get +30 px)
    for label, bb in texts:
        if label.startswith("[tick]"):
            continue
        if (bb.x0 < ax_bbox.x0 - 6 or bb.x1 > ax_bbox.x1 + 6 or
                bb.y0 < ax_bbox.y0 - 6 or bb.y1 > ax_bbox.y1 + 30):
            over = []
            if bb.x0 < ax_bbox.x0 - 6:
                over.append(f"left by {ax_bbox.x0-6-bb.x0:.0f}px")
            if bb.x1 > ax_bbox.x1 + 6:
                over.append(f"right by {bb.x1-ax_bbox.x1-6:.0f}px")
            if bb.y0 < ax_bbox.y0 - 6:
                over.append(f"bottom by {ax_bbox.y0-6-bb.y0:.0f}px")
            if bb.y1 > ax_bbox.y1 + 30:
                over.append(f"top by {bb.y1-ax_bbox.y1-30:.0f}px")
            issues += 1
            print(f"CLIP     [{name}] '{label}'  ->  {', '.join(over)}")

print(f"\n{issues} issue(s) found" if issues else "\nLAYOUT CLEAN: no text overlaps, no clipping")
