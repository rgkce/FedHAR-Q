r"""
FedHAR-Q Sonuç Tablosu (CSV / Markdown / IEEE LaTeX) ve Grafik Üretim Modülü (`code/experiments/export_results.py`).

Bu modül:
1. `result/logs/` altındaki 21 deney koşusunu (`E1`–`E7` × `seeds: [42, 123, 999]`) okur.
2. Her deney koşulu için 3 tohumun **Ortalama ± Standart Sapma (`Mean ± Std`)** özetini:
   - `result/tables/summary_mean_std.csv`
   - `result/tables/summary_table_markdown.md`
   - `result/tables/ieee_results_table.tex`
   dosyalarına yazar.
3. Bildiriye doğrudan eklenecek 4 yüksek çözünürlüklü (300 DPI `.png` ve vektörel `.pdf`)
   akademik grafiği hem `result/plots/` hem de `paper/figures/` dizinlerine kaydeder:
   - `fig1_accuracy_f1_comparison`: E1–E7 Accuracy ve Macro-F1 çubuk grafiği (± std hata çubuklu).
   - `fig2_dp_quantization_interaction`: Gizlilik bütçesi (DP Yok, ε₂=8.0, ε₁=3.0) × Temsil (FP32 vs INT8) etkileşim grafiği.
   - `fig3_convergence_and_epsilon_curves`: Federatif yakınsama eğrileri ve Rényi DP ε bütçe harcama eğrisi.
   - `fig4_size_tradeoff_and_per_class_f1`: Model boyutu–doğruluk ödünleşimi ve 6 aktivite sınıfı bazında F1 ısı haritası (heatmap).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.config_loader import get_path, load_config
from data.preprocessing import ACTIVITY_NAMES

EXPERIMENT_ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]

EXPERIMENT_LABELS = {
    "E1": "E1: Central\n(No DP, FP32)",
    "E2": "E2: FL\n(No DP, FP32)",
    "E3": "E3: FL\n(No DP, INT8)",
    "E4": "E4: FL\n($\\varepsilon_1$=3.0, FP32)",
    "E5": "E5: FL\n($\\varepsilon_1$=3.0, INT8)",
    "E6": "E6: FL\n($\\varepsilon_2$=8.0, FP32)",
    "E7": "E7: FL\n($\\varepsilon_2$=8.0, INT8)",
}


def load_all_experiment_logs(seeds: List[int] | None = None) -> List[Dict[str, Any]]:
    """Tüm deney koşularının JSON kayıtlarını yükler."""
    cfg = load_config()
    if seeds is None:
        seeds = [int(s) for s in cfg["project"]["seeds"]]

    logs: List[Dict[str, Any]] = []
    for exp_id in EXPERIMENT_ORDER:
        for seed in seeds:
            p = get_path(f"result/logs/{exp_id}_seed{seed}.json")
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    logs.append(json.load(f))
    return logs


def generate_summary_tables(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    E1–E7 için 3 tohumun Ortalama ± Standart Sapma tablolarını üretir ve
    CSV, Markdown ve IEEE LaTeX formatlarında kaydeder.
    """
    rows = []
    for r in logs:
        row = {
            "experiment_id": r["experiment_id"],
            "seed": r["seed"],
            "training_mode": r["training_mode"],
            "privacy": r["privacy"],
            "epsilon_target": r.get("epsilon_target"),
            "epsilon_spent": r.get("epsilon_spent"),
            "delta": r.get("delta"),
            "noise_multiplier": r.get("noise_multiplier"),
            "representation": r["representation"],
            "accuracy": float(r["accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "loss": float(r["loss"]),
            "model_size_kb": float(r["model_size_kb"]),
            "latency_mean_ms": float(r["latency_mean_ms"]),
            "peak_inference_footprint_kb": float(r["peak_inference_footprint_kb"]),
            "total_comm_mb": float(r["total_comm_mb"]),
        }
        for act_name, f1_val in r.get("per_class_f1", {}).items():
            row[f"f1_{act_name}"] = float(f1_val)
        rows.append(row)

    df = pd.DataFrame(rows)

    summary_rows = []
    for exp_id in EXPERIMENT_ORDER:
        sub = df[df["experiment_id"] == exp_id]
        if sub.empty:
            continue
        first = sub.iloc[0]

        acc_mean, acc_std = sub["accuracy"].mean(), sub["accuracy"].std(ddof=1)
        f1_mean, f1_std = sub["macro_f1"].mean(), sub["macro_f1"].std(ddof=1)
        size_mean = sub["model_size_kb"].mean()
        lat_mean, lat_std = sub["latency_mean_ms"].mean(), sub["latency_mean_ms"].std(ddof=1)
        mem_mean = sub["peak_inference_footprint_kb"].mean()
        comm_mean = sub["total_comm_mb"].mean()

        eps_spent = first["epsilon_spent"]
        eps_str = f"{eps_spent:.2f}" if pd.notna(eps_spent) and eps_spent is not None else "∞ (Yok)"
        sigma_val = first["noise_multiplier"]
        sigma_str = f"{sigma_val:.2f}" if pd.notna(sigma_val) and sigma_val is not None else "-"

        s_row: Dict[str, Any] = {
            "experiment_id": exp_id,
            "training_mode": first["training_mode"],
            "privacy": first["privacy"],
            "epsilon_spent": eps_str,
            "noise_multiplier": sigma_str,
            "representation": first["representation"],
            "n_seeds": len(sub),
            "acc_mean": round(float(acc_mean), 2),
            "acc_std": round(float(acc_std), 2),
            "accuracy_formatted": f"{acc_mean:.2f} ± {acc_std:.2f}",
            "f1_mean": round(float(f1_mean), 2),
            "f1_std": round(float(f1_std), 2),
            "macro_f1_formatted": f"{f1_mean:.2f} ± {f1_std:.2f}",
            "model_size_kb": round(float(size_mean), 2),
            "latency_mean_ms": round(float(lat_mean), 4),
            "latency_std_ms": round(float(lat_std), 4),
            "latency_formatted": f"{lat_mean:.3f} ± {lat_std:.3f}",
            "peak_memory_kb": round(float(mem_mean), 2),
            "fl_comm_mb": round(float(comm_mean), 2),
        }

        for act_name in ACTIVITY_NAMES.values():
            col = f"f1_{act_name}"
            if col in sub.columns:
                s_row[f"{col}_mean"] = round(float(sub[col].mean()), 2)
                s_row[f"{col}_std"] = round(float(sub[col].std(ddof=1)), 2)

        summary_rows.append(s_row)

    df_summary = pd.DataFrame(summary_rows)
    tables_dir = get_path("result/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV Kaydı
    csv_path = tables_dir / "summary_mean_std.csv"
    df_summary.to_csv(csv_path, index=False)

    # 2. Markdown Raporu Kaydı
    md_lines = [
        "# FedHAR-Q — Deney Matrisi Sonuç Tabloları (3 Tohum Ortalama ± Standart Sapma)",
        "",
        "## Tablo 1: Ana Karşılaştırma Tablosu (E1–E7)",
        "",
        "| Kod | Eğitim Modu | Gizlilik (DP) | Harcanan $\\varepsilon$ ($\\delta=10^{-4}$) | $\\sigma$ | Temsil | Accuracy (%) | Macro-F1 (%) | Model Boyutu (KB) | Çıkarım Gecikmesi (ms) | Tepe Bellek (KB) | FL İletişim (MB) |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_summary.iterrows():
        md_lines.append(
            f"| **{r['experiment_id']}** | {r['training_mode']} | {r['privacy']} | "
            f"{r['epsilon_spent']} | {r['noise_multiplier']} | **{r['representation']}** | "
            f"**{r['accuracy_formatted']}** | **{r['macro_f1_formatted']}** | "
            f"{r['model_size_kb']:.2f} | {r['latency_formatted']} | "
            f"{r['peak_memory_kb']:.2f} | {r['fl_comm_mb']:.2f} |"
        )

    md_lines.extend(
        [
            "",
            "## Tablo 2: Aktivite Sınıfı Bazında F1 Skorları (Ortalama %)",
            "",
            "| Kod | Koşul | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in df_summary.iterrows():
        md_lines.append(
            f"| **{r['experiment_id']}** | {r['privacy']} ({r['representation']}) | "
            f"{r['f1_WALKING_mean']:.2f} | {r['f1_WALKING_UPSTAIRS_mean']:.2f} | "
            f"{r['f1_WALKING_DOWNSTAIRS_mean']:.2f} | {r['f1_SITTING_mean']:.2f} | "
            f"{r['f1_STANDING_mean']:.2f} | {r['f1_LAYING_mean']:.2f} |"
        )

    md_path = tables_dir / "summary_table_markdown.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # 3. IEEE LaTeX Tablosu Kaydı
    tex_lines = [
        "% FedHAR-Q IEEE Konferans Bildirisi Ana Sonuç Tablosu",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{UCI HAR Test Kümesi Üzerinde Diferansiyel Gizlilik ve INT8 Kuantizasyon Sonuçları (3 Tohum Ortalama $\\pm$ Std)}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{llccccccccc}",
        "\\hline",
        "\\textbf{Kod} & \\textbf{Eğitim} & \\textbf{Gizlilik} & $\\boldsymbol{\\varepsilon}$ ($\\delta=10^{-4}$) & $\\boldsymbol{\\sigma}$ & \\textbf{Temsil} & \\textbf{Accuracy (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{Boyut (KB)} & \\textbf{Gecikme (ms)} & \\textbf{İletişim (MB)} \\\\",
        "\\hline",
    ]
    for _, r in df_summary.iterrows():
        eps_tex = f"${r['epsilon_spent']}$" if r["epsilon_spent"] != "∞ (Yok)" else "$\\infty$"
        acc_tex = f"${r['acc_mean']:.2f} \\pm {r['acc_std']:.2f}$"
        f1_tex = f"${r['f1_mean']:.2f} \\pm {r['f1_std']:.2f}$"
        lat_tex = f"${r['latency_mean_ms']:.3f} \\pm {r['latency_std_ms']:.3f}$"
        tex_lines.append(
            f"{r['experiment_id']} & {r['training_mode']} & {r['privacy']} & {eps_tex} & "
            f"{r['noise_multiplier']} & {r['representation']} & {acc_tex} & {f1_tex} & "
            f"{r['model_size_kb']:.2f} & {lat_tex} & {r['fl_comm_mb']:.2f} \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])

    tex_path = tables_dir / "ieee_results_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    return df_summary


def generate_all_plots(logs: List[Dict[str, Any]], df_summary: pd.DataFrame) -> List[Path]:
    """
    Bildiri için 4 adet yüksek çözünürlüklü akademik grafik üretir ve hem
    `result/plots/` hem de `paper/figures/` altına kaydeder.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plots_dir = get_path("result/plots")
    paper_fig_dir = get_path("paper/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    def _save_fig(fig: plt.Figure, basename: str) -> None:
        for folder in (plots_dir, paper_fig_dir):
            png_p = folder / f"{basename}.png"
            pdf_p = folder / f"{basename}.pdf"
            fig.savefig(png_p, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_p, bbox_inches="tight")
            if folder == plots_dir:
                saved_paths.append(png_p)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # ŞEKİL 1: E1-E7 Accuracy ve Macro-F1 Karşılaştırması (Hata Çubuklu)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(df_summary))
    width = 0.36

    bars1 = ax1.bar(
        x - width / 2,
        df_summary["acc_mean"],
        width,
        yerr=df_summary["acc_std"],
        capsize=4,
        label="Test Accuracy (%)",
        color="#1f77b4",
        edgecolor="black",
        alpha=0.9,
    )
    bars2 = ax1.bar(
        x + width / 2,
        df_summary["f1_mean"],
        width,
        yerr=df_summary["f1_std"],
        capsize=4,
        label="Test Macro-F1 (%)",
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.9,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([EXPERIMENT_LABELS[e] for e in df_summary["experiment_id"]])
    ax1.set_ylabel("Başarım (%)")
    ax1.set_ylim(45, 102)
    ax1.set_title(
        "E1–E7 Deney Koşullarında Test Accuracy ve Macro-F1 Karşılaştırması (3 Tohum Ortalama ± Std)",
        fontweight="bold",
    )
    ax1.legend(loc="upper right", frameon=True)

    for b in bars1:
        h = b.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h + 1.2),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    for b in bars2:
        h = b.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h + 1.2),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    _save_fig(fig1, "fig1_accuracy_f1_comparison")

    # -------------------------------------------------------------------------
    # ŞEKİL 2: DP × INT8 Kuantizasyon Etkileşim Analizi (RQ1 & RQ2)
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    regimes = ["Gizliliksiz\n(DP Yok, $\\varepsilon=\\infty$)", "Esnek DP\n($\\varepsilon_2 = 8.0$)", "Sıkı DP\n($\\varepsilon_1 = 3.0$)"]
    fp32_ids = ["E2", "E6", "E4"]
    int8_ids = ["E3", "E7", "E5"]

    df_idx = df_summary.set_index("experiment_id")
    fp32_acc = [df_idx.loc[i, "acc_mean"] for i in fp32_ids]
    fp32_acc_err = [df_idx.loc[i, "acc_std"] for i in fp32_ids]
    int8_acc = [df_idx.loc[i, "acc_mean"] for i in int8_ids]
    int8_acc_err = [df_idx.loc[i, "acc_std"] for i in int8_ids]

    fp32_f1 = [df_idx.loc[i, "f1_mean"] for i in fp32_ids]
    fp32_f1_err = [df_idx.loc[i, "f1_std"] for i in fp32_ids]
    int8_f1 = [df_idx.loc[i, "f1_mean"] for i in int8_ids]
    int8_f1_err = [df_idx.loc[i, "f1_std"] for i in int8_ids]

    rx = np.arange(len(regimes))
    ax2a.errorbar(
        rx, fp32_acc, yerr=fp32_acc_err, marker="o", linewidth=2.2, markersize=8, capsize=5,
        label="FP32 (167.28 KB)", color="#1f77b4"
    )
    ax2a.errorbar(
        rx, int8_acc, yerr=int8_acc_err, marker="s", linestyle="--", linewidth=2.2, markersize=8,
        capsize=5, label="INT8 PTQ (59.75 KB)", color="#d62728"
    )
    ax2a.set_xticks(rx)
    ax2a.set_xticklabels(regimes)
    ax2a.set_ylabel("Test Accuracy (%)")
    ax2a.set_title("(a) Gizlilik Rejimi × Kuantizasyon (Accuracy)", fontweight="bold")
    ax2a.set_ylim(50, 98)
    ax2a.legend(loc="upper right")

    ax2b.errorbar(
        rx, fp32_f1, yerr=fp32_f1_err, marker="o", linewidth=2.2, markersize=8, capsize=5,
        label="FP32 (167.28 KB)", color="#1f77b4"
    )
    ax2b.errorbar(
        rx, int8_f1, yerr=int8_f1_err, marker="s", linestyle="--", linewidth=2.2, markersize=8,
        capsize=5, label="INT8 PTQ (59.75 KB)", color="#d62728"
    )
    ax2b.set_xticks(rx)
    ax2b.set_xticklabels(regimes)
    ax2b.set_ylabel("Test Macro-F1 (%)")
    ax2b.set_title("(b) Gizlilik Rejimi × Kuantizasyon (Macro-F1)", fontweight="bold")
    ax2b.set_ylim(50, 98)
    ax2b.legend(loc="upper right")

    fig2.tight_layout()
    _save_fig(fig2, "fig2_dp_quantization_interaction")

    # -------------------------------------------------------------------------
    # ŞEKİL 3: Federatif Yakınsama (Convergence) ve Rényi DP Bütçe Harcama Eğrisi
    # -------------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    colors = {"E2": "#2ca02c", "E6": "#1f77b4", "E4": "#d62728"}
    titles = {
        "E2": "E2: FL (DP Yok, $\\varepsilon=\\infty$)",
        "E6": "E6: FL (DP $\\varepsilon_2=8.0, \\sigma=2.03$)",
        "E4": "E4: FL (DP $\\varepsilon_1=3.0, \\sigma=4.39$)",
    }

    for exp_id in ("E2", "E6", "E4"):
        exp_logs = [l for l in logs if l["experiment_id"] == exp_id and "history" in l]
        if not exp_logs:
            continue
        rounds = [h["round"] for h in exp_logs[0]["history"]]
        val_f1_matrix = np.array([[h["val_macro_f1"] for h in l["history"]] for l in exp_logs])
        mean_curve = val_f1_matrix.mean(axis=0)
        std_curve = val_f1_matrix.std(axis=0)

        ax3a.plot(rounds, mean_curve, label=titles[exp_id], color=colors[exp_id], linewidth=2.2)
        ax3a.fill_between(
            rounds, mean_curve - std_curve, mean_curve + std_curve, color=colors[exp_id], alpha=0.18
        )

        if exp_id in ("E6", "E4"):
            eps_curve = [h["epsilon_spent"] for h in exp_logs[0]["history"]]
            ax3b.plot(
                rounds, eps_curve, marker=".", label=titles[exp_id], color=colors[exp_id], linewidth=2.2
            )

    ax3a.set_xlabel("Federatif İletişim Turu (Round)")
    ax3a.set_ylabel("Doğrulama (Validation) Macro-F1 (%)")
    ax3a.set_title("(a) Federatif Eğitim Yakınsama Eğrileri (3 Tohum)", fontweight="bold")
    ax3a.set_ylim(25, 102)
    ax3a.legend(loc="lower right")

    ax3b.axhline(8.0, color="#1f77b4", linestyle=":", alpha=0.7, label="Hedef $\\varepsilon_2 = 8.0$")
    ax3b.axhline(3.0, color="#d62728", linestyle=":", alpha=0.7, label="Hedef $\\varepsilon_1 = 3.0$")
    ax3b.set_xlabel("Federatif İletişim Turu (Round)")
    ax3b.set_ylabel("Harcanan Kümülatif Gizlilik Bütçesi ($\\varepsilon$)")
    ax3b.set_title("(b) Rényi DP (RDPAccountant, $\\delta=10^{-4}$) Bütçe Artışı", fontweight="bold")
    ax3b.legend(loc="upper left")

    fig3.tight_layout()
    _save_fig(fig3, "fig3_convergence_and_epsilon_curves")

    # -------------------------------------------------------------------------
    # ŞEKİL 4: Model Boyutu vs Doğruluk Ödünleşimi ve Sınıf Bazlı F1 Isı Haritası
    # -------------------------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.35]})

    markers = {"FP32": "o", "INT8": "D"}
    palette = {
        "E1": "#333333",
        "E2": "#2ca02c",
        "E3": "#2ca02c",
        "E4": "#d62728",
        "E5": "#d62728",
        "E6": "#1f77b4",
        "E7": "#1f77b4",
    }

    for _, r in df_summary.iterrows():
        eid = r["experiment_id"]
        rep = r["representation"]
        ax4a.scatter(
            r["model_size_kb"],
            r["acc_mean"],
            s=140,
            marker=markers[rep],
            color=palette[eid],
            edgecolor="black",
            zorder=5,
            label=f"{eid} ({rep})",
        )
        offset_y = 1.6 if rep == "FP32" else -3.2
        ax4a.annotate(
            f"{eid} ({r['acc_mean']:.1f}%)",
            (r["model_size_kb"], r["acc_mean"]),
            xytext=(0, offset_y * 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax4a.set_xlabel("Model Disk Boyutu (KB)")
    ax4a.set_ylabel("Test Accuracy (%)")
    ax4a.set_xlim(35, 195)
    ax4a.set_ylim(52, 99)
    ax4a.set_title("(a) Model Boyutu (KB) vs. Test Accuracy (%)", fontweight="bold")

    # Sınıf bazlı F1 ısı haritası (Heatmap)
    act_cols = [f"f1_{name}_mean" for name in ACTIVITY_NAMES.values()]
    short_names = ["WALK", "WALK_UP", "WALK_DN", "SITTING", "STANDING", "LAYING"]
    heat_data = df_summary[act_cols].to_numpy()

    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=short_names,
        yticklabels=df_summary["experiment_id"].tolist(),
        cbar_kws={"label": "F1 Skoru (%)"},
        ax=ax4b,
        vmin=35,
        vmax=100,
    )
    ax4b.set_title("(b) Aktivite Sınıfı Bazında F1 Skorları (%)", fontweight="bold")
    ax4b.set_ylabel("Deney Kodu")

    fig4.tight_layout()
    _save_fig(fig4, "fig4_size_tradeoff_and_per_class_f1")

    return saved_paths


def export_all() -> pd.DataFrame:
    logs = load_all_experiment_logs()
    df_summary = generate_summary_tables(logs)
    saved_plots = generate_all_plots(logs, df_summary)
    print(f"[DIŞA AKTARIM TAMAMLANDI] {len(saved_plots)} grafik ve 3 özet tablo üretildi.")
    return df_summary


if __name__ == "__main__":
    export_all()
