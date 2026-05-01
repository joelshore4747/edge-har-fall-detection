#!/usr/bin/env python3
"""Generate the five 'extra' architecture figures with custom per-figure layouts.

Each figure has a layout fitted to its content rather than a shared template:

  Fig 08 - gate-driven pipeline with explicit validation gates between stages
  Fig 09 - vertical trace receipt: claim back to source artefact
  Fig 10 - boundary wall: restricted -> transform -> published
  Fig 11 - sequence diagram with actor lanes and message arrows
  Fig 12 - roadmap timeline: current evidence vs. future MobiAct 2.0 milestones

The PDFs are submission-ready figures. Editable .drawio sources remain in
apps/admin/src/features/architecture/diagrams/ for designer use.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = REPO_ROOT / "docs/figure_exports"
PAGE_W = 1169
PAGE_H = 827

C = {
    "bg": "#FAFAFA",
    "ink": "#1F1B17",
    "muted": "#5A5147",
    "faint": "#9A9081",
    "line": "#D4CFC0",
    "rule": "#EAE3D2",
    "red": "#B4624A",
    "teal": "#2F7F7E",
    "gold": "#C09545",
    "blue": "#4E6E8E",
    "green": "#5C7A47",
    "purple": "#79608A",
    "private_bg": "#FBF1ED",
    "public_bg": "#EEF6F5",
    "neutral_bg": "#F4F0E5",
}


# ---------- shared helpers ---------- #

def setup_page(eyebrow: str, title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(PAGE_H, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), PAGE_W, PAGE_H, facecolor=C["bg"], edgecolor="none"))
    ax.text(60, 50, eyebrow, color=C["faint"], fontsize=8.4, fontweight="bold", va="center")
    ax.text(60, 86, title, color=C["ink"], fontsize=24, fontweight="bold", va="center", family="serif")
    ax.text(60, 116, subtitle, color=C["muted"], fontsize=11, va="center")
    ax.plot([60, 1109], [134, 134], color=C["line"], linewidth=0.8)
    return fig, ax


def add_caption(ax, caption: str) -> None:
    ax.plot([60, 1109], [754, 754], color=C["line"], linewidth=0.8)
    ax.text(60, 768, "\n".join(wrap(caption, width=148)),
            color=C["muted"], fontsize=9, style="italic", va="top", linespacing=1.4)


# ---------- Figure 08 - evaluation protocol ---------- #

def draw_fig08():
    fig, ax = setup_page(
        "EVALUATION VIEW - SUBJECT SPLITS AND CLAIM GATES",
        "Evaluation protocol",
        "Five stages, four validation gates. Every cited metric traces back through them.",
    )

    cx_list = [145, 360, 575, 790, 1005]
    cy = 280
    radius = 46

    stages = [
        ("1", "CORPORA",       "raw datasets",         "blue"),
        ("2", "HARMONISE",     "schema + units",       "teal"),
        ("3", "SUBJECT SPLIT", "group, not random",    "red"),
        ("4", "RUN",           "frozen run_id",        "gold"),
        ("5", "REPORT",        "claim + caveat",       "green"),
    ]

    ax.plot([cx_list[0], cx_list[-1]], [cy, cy], color=C["line"], linewidth=2.5, zorder=1)

    for cx, (num, label, descr, color) in zip(cx_list, stages):
        ax.add_patch(Circle((cx, cy), radius, facecolor="#FFFFFF",
                            edgecolor=C[color], linewidth=2.4, zorder=3))
        ax.text(cx, cy, num, color=C[color], fontsize=26, fontweight="bold",
                ha="center", va="center", family="serif", zorder=4)
        ax.text(cx, cy + radius + 26, label, color=C["ink"], fontsize=10,
                fontweight="bold", ha="center", va="center")
        ax.text(cx, cy + radius + 44, descr, color=C["muted"], fontsize=9.5,
                ha="center", va="center")

    gates = [
        ("schema valid",     "teal"),
        ("subjects grouped", "red"),
        ("run_id frozen",    "gold"),
        ("caveat written",   "green"),
    ]
    for i, (rule, color) in enumerate(gates):
        gx = (cx_list[i] + cx_list[i + 1]) / 2
        size = 11
        diamond = Polygon(
            [(gx, cy - size), (gx + size, cy), (gx, cy + size), (gx - size, cy)],
            facecolor="#FFFFFF", edgecolor=C[color], linewidth=1.8, zorder=4,
        )
        ax.add_patch(diamond)
        ax.text(gx, cy - size - 14, "GATE", color=C[color], fontsize=7.5,
                fontweight="bold", ha="center", va="center")
        ax.text(gx, cy + size + 16, rule, color=C[color], fontsize=8.5,
                ha="center", va="top", style="italic")

    chip_y = 480
    ax.text(60, chip_y - 24, "CORPORA IN SCOPE", color=C["faint"],
            fontsize=8.4, fontweight="bold", va="center")

    chips = [
        ("UCI HAR",      "blue",   False),
        ("PAMAP2",       "blue",   False),
        ("WISDM*",       "blue",   True),
        ("MobiFall",     "blue",   False),
        ("SisFall",      "blue",   False),
        ("PHONE1†", "purple", True),
        ("MobiAct 2.0‡", "faint", True),
    ]
    chip_x = 60
    for label, color, dashed in chips:
        text_w = len(label) * 8.5 + 22
        ax.add_patch(FancyBboxPatch(
            (chip_x, chip_y - 14), text_w, 28,
            boxstyle="round,pad=0,rounding_size=14",
            facecolor="#FFFFFF", edgecolor=C[color], linewidth=1.3,
            linestyle="--" if dashed else "-",
        ))
        ax.text(chip_x + text_w / 2, chip_y, label, color=C[color],
                fontsize=9.5, fontweight="bold", ha="center", va="center")
        chip_x += text_w + 12

    ax.text(60, chip_y + 38,
            "*  WISDM has no clean subject IDs - flagged in caveat.    "
            "†  PHONE1 is a runtime probe, not a headline corpus.    "
            "‡  MobiAct 2.0 is approved future work; not in submitted metrics.",
            color=C["muted"], fontsize=8.5, va="top")

    rule_y = 590
    ax.add_patch(FancyBboxPatch((60, rule_y), 1049, 110,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor=C["neutral_bg"], edgecolor=C["line"], linewidth=1))
    ax.text(80, rule_y + 22, "REPORTING RULE", color=C["gold"],
            fontsize=8.4, fontweight="bold", va="center")
    ax.text(80, rule_y + 54,
            "A metric is cited only when it has a source artefact.",
            color=C["ink"], fontsize=14, fontweight="bold", va="center", family="serif")
    ax.text(80, rule_y + 84,
            "Headline metrics: macro-F1 / F1, bootstrap CI, paired tests where valid.    "
            "New corpora must clear all four gates before they may support a claim.",
            color=C["muted"], fontsize=9.5, va="center")

    add_caption(ax,
        "Evaluation protocol used to protect the dissertation results from subject leakage and unaudited late evidence. "
        "The diamond markers are validation gates that must pass before the next stage; new datasets enter the pipeline only "
        "after every gate has cleared."
    )
    return fig


# ---------- Figure 09 - evidence provenance ---------- #

def draw_fig09():
    fig, ax = setup_page(
        "REPRODUCIBILITY VIEW - CLAIM TRACEABILITY",
        "Evidence provenance",
        "Every cited metric traces from the chapter back to a source artefact regenerable from a script.",
    )

    card_x, card_y, card_w, card_h = 60, 158, 1049, 64
    ax.add_patch(FancyBboxPatch((card_x, card_y), card_w, card_h,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor=C["neutral_bg"], edgecolor=C["line"], linewidth=1))
    ax.text(card_x + 20, card_y + 18, "EXAMPLE CLAIM", color=C["gold"],
            fontsize=8.4, fontweight="bold", va="center")
    ax.text(card_x + 20, card_y + 44,
            "“Macro-F1 = 0.83 on UCI HAR with subject-independent split.”   "
            "→   cited in Chapter 8, Table 8.3.",
            color=C["ink"], fontsize=12, fontweight="bold", va="center", family="serif")

    rows = [
        ("1", "SOURCE",    "data/raw/uci_har/",
                           "origin file, downloaded once",                 "blue"),
        ("2", "HARMONISE", "data/processed/uci_har/windows.parquet",
                           "schema + m/s² + 50 Hz",                    "teal"),
        ("3", "FEATURES",  "data/processed/uci_har/features.parquet",
                           "schema-checked window stats",                   "green"),
        ("4", "RUN",       "results/runs/<run_id>/{args.json, git_sha.txt, predictions.csv}",
                           "frozen run id + commit",                        "gold"),
        ("5", "ARTEFACT",  "results/runs/<run_id>/metrics.json + figures/",
                           "regenerable from scripts/",                     "red"),
        ("6", "CLAIM",     "docs/dissertation/08_evaluation_results_and_discussion.md",
                           "cited with caveat in the chapter",              "purple"),
    ]

    row_y0 = 248
    row_h = 44
    ax.plot([60, 1109], [row_y0, row_y0], color=C["rule"], linewidth=0.6)
    for i, (num, stage, path, why, color) in enumerate(rows):
        y = row_y0 + i * row_h
        ax.text(80, y + row_h / 2, num, color=C[color], fontsize=18,
                fontweight="bold", ha="center", va="center", family="serif")
        ax.text(118, y + row_h / 2, stage, color=C[color], fontsize=9.5,
                fontweight="bold", va="center")
        ax.text(282, y + row_h / 2, path, color=C["ink"], fontsize=9.5,
                family="monospace", va="center")
        ax.text(820, y + row_h / 2, why, color=C["muted"], fontsize=9.5,
                va="center", style="italic")
        ax.plot([60, 1109], [y + row_h, y + row_h], color=C["rule"], linewidth=0.6)

    panel_y = row_y0 + len(rows) * row_h + 20
    panel_h = 116
    panel_w = 510

    ax.add_patch(FancyBboxPatch((60, panel_y), panel_w, panel_h,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#FFFFFF", edgecolor=C["teal"], linewidth=1.2))
    ax.text(80, panel_y + 22, "AUDIT QUESTIONS", color=C["teal"],
            fontsize=8.4, fontweight="bold", va="center")
    audit_lines = [
        "✓  Can the source file be named?",
        "✓  Can the run_id and git SHA be named?",
        "✓  Can a script regenerate the artefact?",
        "✓  Is the caveat present in the chapter?",
    ]
    for j, line in enumerate(audit_lines):
        ax.text(80, panel_y + 46 + j * 18, line, color=C["ink"], fontsize=10, va="center")

    ax.add_patch(FancyBboxPatch((598, panel_y), panel_w, panel_h,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#FFFFFF", edgecolor=C["red"], linewidth=1.2))
    ax.text(618, panel_y + 22, "STOP CONDITIONS", color=C["red"],
            fontsize=8.4, fontweight="bold", va="center")
    stop_lines = [
        "✗  Source artefact missing.",
        "✗  Hand-authored number with no provenance.",
        "✗  Metric changed without regenerated figure.",
        "✗  Private evidence exposed in published artefact.",
    ]
    for j, line in enumerate(stop_lines):
        ax.text(618, panel_y + 46 + j * 18, line, color=C["ink"], fontsize=10, va="center")

    add_caption(ax,
        "Evidence provenance chain used to make the dissertation auditable. The intended examiner path is from any cited "
        "figure or metric back to a run artefact, script, harmonised data source, and limitation statement."
    )
    return fig


# ---------- Figure 10 - privacy and ethics boundary ---------- #

def draw_fig10():
    fig, ax = setup_page(
        "ETHICS VIEW - PRIVATE DATA VERSUS PUBLISHED EVIDENCE",
        "Privacy and ethics boundary",
        "What stays restricted, how it is transformed, and what may appear in dissertation artefacts.",
    )

    col_y = 168
    col_h = 444

    lx, lw = 60, 360
    ax.add_patch(FancyBboxPatch((lx, col_y), lw, col_h,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor=C["private_bg"], edgecolor=C["red"], linewidth=1.4))
    ax.text(lx + 22, col_y + 26, "RESTRICTED", color=C["red"],
            fontsize=8.4, fontweight="bold", va="center")
    ax.text(lx + 22, col_y + 64, "Stays off-repo", color=C["ink"],
            fontsize=18, fontweight="bold", va="center", family="serif")
    items_l = [
        "Signed consent forms",
        "Raw IMU payloads (per-subject)",
        "login_username rows in app DB",
        "MobiAct 2.0 access email + link",
        "Database backups",
    ]
    for i, item in enumerate(items_l):
        ax.text(lx + 22, col_y + 124 + i * 44, item, color=C["ink"],
                fontsize=11, va="center")

    mx, mw = 470, 230
    ax.add_patch(FancyBboxPatch((mx, col_y + 60), mw, col_h - 120,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor="#FFFFFF", edgecolor=C["gold"], linewidth=1.4))
    ax.text(mx + mw / 2, col_y + 80, "TRANSFORM", color=C["gold"],
            fontsize=8.4, fontweight="bold", ha="center", va="center")
    ax.text(mx + mw / 2, col_y + 110, "Publication", color=C["ink"],
            fontsize=15, fontweight="bold", ha="center", va="center", family="serif")
    ax.text(mx + mw / 2, col_y + 132, "controls", color=C["ink"],
            fontsize=15, fontweight="bold", ha="center", va="center", family="serif")

    transforms = ["Opaque ID hash", "Aggregation + CI", "Redaction"]
    for i, t in enumerate(transforms):
        py = col_y + 200 + i * 50
        ax.add_patch(FancyBboxPatch((mx + 18, py - 14), mw - 36, 28,
                                    boxstyle="round,pad=0,rounding_size=14",
                                    facecolor=C["neutral_bg"], edgecolor=C["gold"], linewidth=1))
        ax.text(mx + mw / 2, py, t, color=C["ink"], fontsize=10.5,
                fontweight="bold", ha="center", va="center")

    rx, rw = 749, 360
    ax.add_patch(FancyBboxPatch((rx, col_y), rw, col_h,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor=C["public_bg"], edgecolor=C["teal"], linewidth=1.4))
    ax.text(rx + 22, col_y + 26, "PUBLISHED", color=C["teal"],
            fontsize=8.4, fontweight="bold", va="center")
    ax.text(rx + 22, col_y + 64, "Dissertation-safe", color=C["ink"],
            fontsize=18, fontweight="bold", va="center", family="serif")
    items_r = [
        "Aggregate metrics + bootstrap CI",
        "Figures and tables (regenerable)",
        "Model card with caveats",
        "Redacted appendix screenshots",
        "Source code + schemas",
    ]
    for i, item in enumerate(items_r):
        ax.text(rx + 22, col_y + 124 + i * 44, item, color=C["ink"],
                fontsize=11, va="center")

    arrow_y = col_y + col_h / 2
    ax.add_patch(FancyArrowPatch((lx + lw + 6, arrow_y), (mx - 6, arrow_y),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2.2, color=C["red"]))
    ax.text((lx + lw + mx) / 2, arrow_y - 18, "minimise",
            color=C["red"], fontsize=10, fontweight="bold",
            ha="center", va="center", style="italic")

    ax.add_patch(FancyArrowPatch((mx + mw + 6, arrow_y), (rx - 6, arrow_y),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2.2, color=C["teal"]))
    ax.text((mx + mw + rx) / 2, arrow_y - 18, "publish",
            color=C["teal"], fontsize=10, fontweight="bold",
            ha="center", va="center", style="italic")

    rule_y = 638
    ax.add_patch(FancyBboxPatch((60, rule_y), 1049, 76,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#FFFFFF", edgecolor=C["ink"], linewidth=1))
    ax.text(80, rule_y + 22, "MODEL INPUT RULE", color=C["ink"],
            fontsize=8.4, fontweight="bold", va="center")
    ax.text(80, rule_y + 52,
            "Models receive IMU + placement features only.   "
            "No demographics.   No login_username.   Subject IDs are opaque in exported results.",
            color=C["ink"], fontsize=11, va="center", family="serif")

    add_caption(ax,
        "Privacy and ethics boundary for the submitted prototype. Restricted records may support the research process, but only "
        "redacted, aggregated, or opaque evidence crosses the boundary into dissertation artefacts."
    )
    return fig


# ---------- Figure 11 - mobile session lifecycle (sequence diagram) ---------- #

def draw_fig11():
    fig, ax = setup_page(
        "RUNTIME VIEW - FROM CONSENTED PHONE RECORDING TO REVIEWABLE EVENT",
        "Mobile session lifecycle",
        "One labelled recording moves from consented user, through upload and inference, into review.",
    )

    actors = [
        ("USER",  "consented",          "blue"),
        ("PHONE", "labelled session",   "teal"),
        ("API",   "/v1/infer/session",  "red"),
        ("MODEL", "HAR + fall + score", "gold"),
        ("DB",    "evidence tables",    "green"),
        ("UI",    "admin review",       "purple"),
    ]

    n = len(actors)
    margin_x = 80
    span = PAGE_W - 2 * margin_x
    actor_xs = [margin_x + (i + 0.5) * span / n for i in range(n)]

    head_y = 162
    head_h = 50
    head_w = span / n - 30
    for (label, sub, color), x in zip(actors, actor_xs):
        ax.add_patch(FancyBboxPatch((x - head_w / 2, head_y), head_w, head_h,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    facecolor="#FFFFFF", edgecolor=C[color], linewidth=1.6))
        ax.text(x, head_y + 18, label, color=C[color], fontsize=10.5,
                fontweight="bold", ha="center", va="center")
        ax.text(x, head_y + 36, sub, color=C["muted"], fontsize=8.5,
                ha="center", va="center", style="italic")

    life_y0 = head_y + head_h + 8
    life_y1 = 600
    for x in actor_xs:
        ax.plot([x, x], [life_y0, life_y1], color=C["faint"],
                linewidth=0.8, linestyle=(0, (3, 3)))

    messages = [
        (0, 1, "1. start session",            "consent + opaque user_id"),
        (1, 1, "2. record",                   "accel/gyro + labels"),
        (1, 2, "3. POST /v1/infer/session",   "windows + request_id"),
        (2, 2, "4. validate",                 "schema, idempotent on request_id"),
        (2, 3, "5. score",                    "HAR head, fall prob, group, vulnerability"),
        (3, 4, "6. write evidence",           "app_session_inferences · app_timeline_events · app_grouped_fall_events"),
        (4, 5, "7. read for review",          "timeline + grouped events"),
        (5, 4, "8. feedback (optional)",      "app_feedback row"),
    ]

    msg_y0 = life_y0 + 22
    msg_dy = 50
    for i, (a, b, label, sub) in enumerate(messages):
        y = msg_y0 + i * msg_dy
        x1, x2 = actor_xs[a], actor_xs[b]
        if a == b:
            ax.text(x1 + 4, y, "↻", color=C["ink"], fontsize=14,
                    ha="left", va="center")
            ax.text(x1 + 22, y - 8, label, color=C["ink"], fontsize=9.5,
                    fontweight="bold", va="center")
            ax.text(x1 + 22, y + 8, sub, color=C["muted"], fontsize=8.5,
                    va="center", style="italic")
        else:
            ax.add_patch(FancyArrowPatch((x1, y), (x2, y),
                                         arrowstyle="-|>", mutation_scale=12,
                                         linewidth=1.6, color=C["ink"]))
            mx = (x1 + x2) / 2
            ax.text(mx, y - 8, label, color=C["ink"], fontsize=9.5,
                    fontweight="bold", ha="center", va="bottom")
            ax.text(mx, y + 8, sub, color=C["muted"], fontsize=8.5,
                    ha="center", va="top", style="italic")

    fm_y = 622
    ax.add_patch(FancyBboxPatch((60, fm_y), 1049, 90,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor="#FFFFFF", edgecolor=C["red"], linewidth=1))
    ax.text(80, fm_y + 22, "FAILURE MODES & DEPLOYMENT GAPS",
            color=C["red"], fontsize=8.4, fontweight="bold", va="center")
    ax.text(80, fm_y + 50,
            "Invalid schema is rejected at step 4.   "
            "Duplicate request_id is idempotent.   "
            "Inference is server-side - acceptable for research, not consumer deployment.",
            color=C["ink"], fontsize=10.5, va="center")
    ax.text(80, fm_y + 72,
            "Future product path: on-device inference, emergency-contact workflow, battery-impact study, user export/delete.",
            color=C["muted"], fontsize=9.5, va="center", style="italic")

    add_caption(ax,
        "Lifecycle of a recorded mobile session from consented user context to persisted inference evidence and admin review. "
        "Database table names match the application schema in Chapter 10."
    )
    return fig


# ---------- Figure 12 - future dataset integration roadmap ---------- #

def draw_fig12():
    fig, ax = setup_page(
        "FUTURE WORK VIEW - APPROVED DATA DOES NOT BECOME A CLAIM UNTIL RERUN",
        "Future dataset integration: MobiAct 2.0",
        "Approved 2026-04-17 · HMU BMI Lab · not in submitted metrics. Four milestones must close before any claim.",
    )

    axis_y = 360
    axis_x0, axis_x1 = 80, 1100
    ax.plot([axis_x0, axis_x1], [axis_y, axis_y], color=C["faint"], linewidth=2)

    now_x = 380
    ax.plot([now_x, now_x], [axis_y - 130, axis_y + 110], color=C["ink"],
            linewidth=1.8, linestyle=(0, (4, 4)))
    ax.text(now_x, axis_y - 150, "SUBMISSION FROZEN", color=C["ink"],
            fontsize=8.4, fontweight="bold", ha="center", va="center")
    ax.text(now_x, axis_y - 134, "2026-04-30", color=C["muted"],
            fontsize=9, ha="center", va="bottom")

    past_x, past_w = 80, now_x - 100
    past_h = 220
    past_y = axis_y - past_h / 2
    ax.add_patch(FancyBboxPatch((past_x, past_y), past_w, past_h,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor=C["public_bg"], edgecolor=C["green"], linewidth=1.4))
    ax.text(past_x + 22, past_y + 24, "CURRENT EVIDENCE", color=C["green"],
            fontsize=8.4, fontweight="bold", va="center")
    ax.text(past_x + 22, past_y + 58, "Frozen corpora", color=C["ink"],
            fontsize=15, fontweight="bold", va="center", family="serif")
    corpora = [
        "UCI HAR",
        "PAMAP2",
        "WISDM",
        "MobiFall",
        "SisFall",
        "PHONE1 (runtime probe)",
    ]
    for i, c in enumerate(corpora):
        ax.text(past_x + 22, past_y + 94 + i * 20, "·  " + c,
                color=C["ink"], fontsize=10.5, va="center")

    ax.add_patch(Circle((past_x + past_w - 24, axis_y), 8,
                        facecolor=C["green"], edgecolor=C["green"], linewidth=2))
    ax.text(past_x + past_w - 24, axis_y + 24, "cited", color=C["green"],
            fontsize=8.5, ha="center", va="center", style="italic")

    milestones = [
        ("M1", "Loader",        "MOBIACT_V2 ingest +\ncanonical labels", "teal"),
        ("M2", "Audit",         "schema check +\nreadiness report",      "blue"),
        ("M3", "Subject split", "group split + run +\nfrozen run_id",    "gold"),
        ("M4", "Republish",     "regenerate metrics,\nfigures, caveats", "red"),
    ]

    n_ms = len(milestones)
    fut_x0 = now_x + 80
    fut_x1 = axis_x1 - 60
    fut_xs = [fut_x0 + (i + 0.5) * (fut_x1 - fut_x0) / n_ms for i in range(n_ms)]

    for x, (mid, title, descr, color) in zip(fut_xs, milestones):
        ax.add_patch(Circle((x, axis_y), 9, facecolor="#FFFFFF",
                            edgecolor=C[color], linewidth=2.4, zorder=4))
        box_w, box_h = 168, 116
        box_y = axis_y - box_h - 32
        ax.add_patch(FancyBboxPatch((x - box_w / 2, box_y), box_w, box_h,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    facecolor="#FFFFFF", edgecolor=C[color], linewidth=1.4))
        ax.plot([x, x], [box_y + box_h, axis_y - 9], color=C[color],
                linewidth=1.0, linestyle=(0, (2, 3)))
        ax.text(x, box_y + 18, mid, color=C[color], fontsize=8.4,
                fontweight="bold", ha="center", va="center")
        ax.text(x, box_y + 46, title, color=C["ink"], fontsize=13.5,
                fontweight="bold", ha="center", va="center", family="serif")
        ax.text(x, box_y + 84, descr, color=C["muted"], fontsize=9.5,
                ha="center", va="center", linespacing=1.3)
        ax.text(x, axis_y + 24, "open", color=C[color], fontsize=8.5,
                ha="center", va="center", style="italic")

    ax.text(now_x - 14, axis_y + 64, "← cited in submission",
            color=C["green"], fontsize=10, ha="right", va="center", style="italic")
    ax.text(now_x + 14, axis_y + 64, "future work - not a claim until all four close →",
            color=C["red"], fontsize=10, ha="left", va="center", style="italic")

    rule_y = 622
    ax.add_patch(FancyBboxPatch((60, rule_y), 1049, 90,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor=C["neutral_bg"], edgecolor=C["line"], linewidth=1))
    ax.text(80, rule_y + 22, "SUCCESS CRITERION", color=C["gold"],
            fontsize=8.4, fontweight="bold", va="center")
    ax.text(80, rule_y + 54,
            "A claim may cite MobiAct 2.0 only after every milestone closes and "
            "the figure/table register points to regenerated artefacts.",
            color=C["ink"], fontsize=12, fontweight="bold", va="center", family="serif")

    add_caption(ax,
        "Future integration path for MobiAct 2.0. The 2026-04-17 access approval is useful provenance, but the dataset cannot "
        "support reported claims until the four milestones close: loader, audit, subject-independent split with frozen run_id, "
        "and regenerated metrics with updated caveats."
    )
    return fig


# ---------- driver ---------- #

def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    items = [
        ("figure_8_2_evaluation_protocol.pdf",        draw_fig08),
        ("figure_7_3_evidence_provenance.pdf",        draw_fig09),
        ("figure_9_1_privacy_ethics_boundary.pdf",    draw_fig10),
        ("figure_6_6_mobile_session_lifecycle.pdf",   draw_fig11),
        ("figure_12_1_future_dataset_integration.pdf", draw_fig12),
    ]

    individual_paths: list[Path] = []
    for name, draw_fn in items:
        out = PDF_DIR / name
        with PdfPages(out) as pdf:
            fig = draw_fn()
            pdf.savefig(fig)
            plt.close(fig)
        individual_paths.append(out)

    combined = PDF_DIR / "extra_architecture_diagrams.pdf"
    with PdfPages(combined) as pdf:
        for _, draw_fn in items:
            fig = draw_fn()
            pdf.savefig(fig)
            plt.close(fig)

    for path in individual_paths + [combined]:
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())