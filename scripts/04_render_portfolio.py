"""Refresh three portfolio figures from committed reports; never fit models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
BLUE, TEAL, GRAY = "#2563EB", "#0F766E", "#64748B"
LABELS = {"s_learner": "S-Learner (selected)", "t_learner": "T-Learner",
          "x_learner": "X-Learner", "causal_forest": "Causal Forest",
          "response_model": "Response baseline"}


def load(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def finish(fig, ax, name, note):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.22, top=0.83)
    fig.text(0.06, 0.07, note, fontsize=9, color=GRAY)
    fig.savefig(ROOT / "figures" / name, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#334155", "text.color": "#0F172A"})
    (ROOT / "figures").mkdir(exist_ok=True)
    m, a, ate = load("uplift_metrics.json"), load("experiment_validation.json"), load("ate_analysis.json")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Causal targeting vs. response prediction", x=0.06, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.06, 0.89, f"Held-out test · {m['model_rows']['test']:,} records · primary outcome: visit", color=GRAY)
    # Muted predeclared methods remain visible; selected policy and baseline stand out.
    order = [n for n in m["test"] if n not in (m["selected_model"], "response_model")]
    order += ["response_model", m["selected_model"]]
    for name in order:
        rows = m["test"][name]["curve"]
        selected = name == m["selected_model"]
        baseline = name == "response_model"
        ax.plot([0] + [r["fraction"] for r in rows],
                [0] + [r["incremental_per_population"] * m["model_rows"]["test"] for r in rows],
                label=LABELS[name], color=BLUE if selected else TEAL if baseline else "#CBD5E1",
                lw=2.8 if selected or baseline else 1.2,
                ls="--" if baseline else "-")
    overall = m["test"][m["selected_model"]]["curve"][-1]["incremental_per_population"] * m["model_rows"]["test"]
    ax.plot([0, 1], [0, overall], color=GRAY, lw=1.3, ls=":", label="Random-targeting reference")
    ax.axvline(0.1, color=GRAY, alpha=0.45, lw=1, ls=":")
    ax.set(xlabel="Share of audience targeted", ylabel="Estimated cumulative incremental visits", xlim=(0, 1), ylim=(0, None))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    finish(fig, ax, "targeting_policy.png", "Propensity-adjusted offline estimate; not a production forecast.\nS-Learner selected on validation Qini, not test performance.")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Advertising increased observed outcome rates", x=0.06, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.89, "Full-sample assigned-treatment effect · 13,979,592 records", color=GRAY)
    for y, name in enumerate(["conversion", "visit"]):
        o = ate["outcomes"][name]
        point, low, high = o["ate"] * 100, o["ci95_normal"][0] * 100, o["ci95_normal"][1] * 100
        ax.errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o", ms=8, color=BLUE,
                    capsize=5, elinewidth=2)
        ax.annotate(f"+{point:.4f} pp", (point, y), xytext=(0, 18), textcoords="offset points", ha="center", color=BLUE)
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set(yticks=[0, 1], yticklabels=["Conversion", "Visit"],
           xlabel="Average treatment effect (percentage points)", xlim=(-0.05, 1.22), ylim=(-0.55, 1.65))
    finish(fig, ax, "ate_confidence_interval.png", "Error bars: 95% normal-approximation confidence intervals.\nHeadline bootstrap intervals are recorded separately; sample effects are not population campaign lift.")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Baseline feature balance before causal modeling", x=0.06, ha="left", fontsize=18, fontweight="bold")
    rows = sorted(a["feature_balance"], key=lambda r: abs(r["smd"]))
    maximum = max(abs(r["smd"]) for r in rows)
    fig.text(0.06, 0.89, f"Maximum absolute SMD: {maximum:.4f} · treatment prediction AUC: {m['assignment_predictability_validation_auc']:.4f}", color=GRAY)
    ax.barh([r["feature"] for r in rows], [r["smd"] for r in rows], color=TEAL, height=0.6)
    ax.axvline(0, color=GRAY, lw=1)
    limit = maximum * 1.4
    ax.set(xlabel="Signed standardized mean difference (treatment − control)", xlim=(-limit, limit))
    finish(fig, ax, "covariate_balance.png", "Full-sample diagnostic using baseline f0–f11; post-treatment exposure excluded.\nBalance and near-chance predictability alone do not prove conditional randomization.")
    print("Refreshed three portfolio figures without modifying reports or models.")


if __name__ == "__main__":
    main()
