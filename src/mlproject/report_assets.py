"""
Generate LaTeX tables and copy figures for report/report.tex from the real
pipeline outputs (data/splits/*.json, outputs/models/*/history.json,
outputs/results/*_results.json). Nothing in the report's results section is typed
by hand: re-run `python -m mlproject report-assets` after every experiment.

Writes report/generated/*.tex and report/figures/*.png
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

from mlproject.config import ProjectConfig, default_config

PRETTY = {
    "custom_cnn": "Custom CNN (scratch)",
    "resnet18": "ResNet-18 (transfer)",
    "vit_b_16": "ViT-B/16 (transfer)",
    "mobilenet_v3": "MobileNetV3-L (transfer)",
}


def _tex(s) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _load(p: Path):
    return json.load(open(p)) if p.exists() else None


def _write(out_dir: Path, name: str, body: str) -> None:
    (out_dir / f"{name}.tex").write_text(body)


def generate(models: List[str] = None, config: ProjectConfig = default_config) -> Path:
    models = models or config.model_families
    report_dir = config.project_root / "report"
    gen = report_dir / "generated"
    figs = report_dir / "figures"
    gen.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    meta = _load(config.splits_dir / "classes.json")
    clean = _load(config.splits_dir / "cleaning_report.json")
    hists = {m: _load(config.models_dir / m / "history.json") for m in models}
    results = {m: _load(config.results_dir / f"{m}_results.json") for m in models}

    # ---- warning banner if results come from synthetic data or are missing
    banner = ""
    if meta is None or meta.get("synthetic_demo_data"):
        banner = (r"\begin{center}\fcolorbox{red}{red!8}{\parbox{0.92\linewidth}{\textbf{\textcolor{red}{DRAFT:}} "
                  r"the numbers in this report come from SYNTHETIC demo data or are missing. "
                  r"Re-run the pipeline on the real dataset before submission.}}\end{center}")
    _write(gen, "banner", banner)

    # ---- macros used in running text
    macros = []
    if meta:
        sc = meta["split_counts"]
        macros += [
            rf"\newcommand{{\NumClasses}}{{{len(meta['classes'])}}}",
            rf"\newcommand{{\NumImages}}{{{sc['total']}}}",
            rf"\newcommand{{\NumTrain}}{{{sc['train']}}}",
            rf"\newcommand{{\NumVal}}{{{sc['val']}}}",
            rf"\newcommand{{\NumTest}}{{{sc['test']}}}",
            rf"\newcommand{{\Seed}}{{{meta['seed']}}}",
            rf"\newcommand{{\SplitMethod}}{{{_tex(meta['split_method'])}}}",
            rf"\newcommand{{\NumGroups}}{{{sum(meta['group_counts'].values())}}}",
        ]
    if clean:
        macros += [
            rf"\newcommand{{\NumRaw}}{{{clean['initial_samples']}}}",
            rf"\newcommand{{\NumCorrupt}}{{{clean['removed_corrupt_or_small']}}}",
            rf"\newcommand{{\NumManual}}{{{clean.get('removed_manual_review', 0)}}}",
            rf"\newcommand{{\NumExactDup}}{{{clean['removed_exact_duplicates']}}}",
            rf"\newcommand{{\NumNearDupConflict}}{{{clean['removed_cross_class_near_duplicates']}}}",
            rf"\newcommand{{\NumNearDupGrouped}}{{{clean['same_class_near_duplicate_pairs_grouped']}}}",
            rf"\newcommand{{\NearDupBits}}{{{clean['near_dup_threshold_bits']}}}",
        ]
    env = next((h["environment"] for h in hists.values() if h), None)
    if env:
        gpu = env["gpu_name"] if env["gpu_name"] != "None" else "none (CPU only)"
        mem = f", {env['gpu_memory_gb']} GB" if env.get("gpu_memory_gb") else ""
        macros += [
            rf"\newcommand{{\EnvPython}}{{{env['python']}}}",
            rf"\newcommand{{\EnvTorch}}{{{_tex(env['torch'])}}}",
            rf"\newcommand{{\EnvTorchvision}}{{{_tex(env['torchvision'])}}}",
            rf"\newcommand{{\EnvSklearn}}{{{env['scikit_learn']}}}",
            rf"\newcommand{{\EnvGPU}}{{{_tex(gpu + mem)}}}",
            rf"\newcommand{{\EnvCPU}}{{{_tex(env['cpu'])}}}",
            rf"\newcommand{{\EnvOS}}{{{_tex(env['os'])}}}",
        ]
    best = None
    done = {m: r for m, r in results.items() if r}
    if done:
        best = max(done, key=lambda m: done[m]["macro_f1"])
        macros += [
            rf"\newcommand{{\BestModel}}{{{_tex(PRETTY.get(best, best))}}}",
            rf"\newcommand{{\BestKey}}{{{best}}}",
            rf"\newcommand{{\BestMacroF}}{{{done[best]['macro_f1']:.3f}}}",
            rf"\newcommand{{\BestAcc}}{{{100 * done[best]['test_accuracy']:.1f}\%}}",
        ]
        for m, r in done.items():
            key = {"custom_cnn": "Cnn", "resnet18": "Resnet", "vit_b_16": "Vit", "mobilenet_v3": "Mobilenet"}.get(m, m)
            macros += [rf"\newcommand{{\Acc{key}}}{{{100 * r['test_accuracy']:.1f}\%}}",
                       rf"\newcommand{{\MacroF{key}}}{{{r['macro_f1']:.3f}}}"]
    _write(gen, "macros", "\n".join(macros) + "\n")

    # ---- dataset table: before/after cleaning + per split
    if meta and clean:
        rows = []
        for c in meta["classes"]:
            pc = meta["per_class_counts"]
            t, v, te = pc["train"].get(c, 0), pc["val"].get(c, 0), pc["test"].get(c, 0)
            rows.append(f"{_tex(c)} & {clean['per_class_before'].get(c, 0)} & {clean['per_class_after'].get(c, 0)} & {t} & {v} & {te} \\\\")
        sc = meta["split_counts"]
        rows.append(r"\midrule")
        rows.append(f"\\textbf{{Total}} & {clean['initial_samples']} & {clean['cleaned_samples']} & {sc['train']} & {sc['val']} & {sc['test']} \\\\")
        _write(gen, "dataset_table", "\n".join([
            r"\begin{table}[H]\centering\small",
            r"\caption{Images per class before and after cleaning, and per split (seed \Seed).}\label{tab:dataset}",
            r"\begin{tabular}{lrrrrr}\toprule",
            r"Class & Raw & Cleaned & Train & Val & Test \\\midrule",
            *rows,
            r"\bottomrule\end{tabular}\end{table}",
        ]))
    else:
        _write(gen, "dataset_table", r"\textit{[Dataset table: run \texttt{python -m mlproject prepare}.]}")

    # ---- hyper-parameter table
    hp_rows = []
    for m in models:
        h = hists.get(m)
        if not h:
            continue
        hp = h["hyperparameters"]
        hp_rows.append(
            f"{_tex(PRETTY.get(m, m))} & {_tex(hp['pretrained_weights'])} & {hp['batch_size']}$\\times${hp['accum_steps']} & "
            f"{hp['max_epochs']} ({h['epochs_trained']}) & {hp['lr']:.0e} & {hp['weight_decay']:.0e} & "
            f"{h['parameters']['total_parameters'] / 1e6:.2f} \\\\")
    _write(gen, "hyperparams_table", "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Training configuration per model. All models: AdamW, input $224\times224$, linear warm-up then cosine decay, "
        r"class-weighted cross-entropy, early stopping on validation loss, seed \Seed. Batch is given as batch$\times$accumulation steps "
        r"(effective batch 32 for all). Epochs: maximum (actually trained).}\label{tab:hparams}",
        r"\begin{tabular}{p{3.1cm}p{3.2cm}cccc r}\toprule",
        r"Model & Pretrained weights & Batch & Epochs & LR & WD & Params (M) \\\midrule",
        *(hp_rows or [r"\multicolumn{7}{c}{\textit{[run training]}}\\"]),
        r"\bottomrule\end{tabular}\end{table}",
    ]))

    # ---- main results table
    res_rows = []
    for m in models:
        r = results.get(m)
        h = hists.get(m)
        if not r:
            continue
        res_rows.append(
            f"{_tex(PRETTY.get(m, m))} & {100 * r['test_accuracy']:.1f} & {100 * r['balanced_accuracy']:.1f} & "
            f"{r['macro_precision']:.3f} & {r['macro_recall']:.3f} & {r['macro_f1']:.3f} & {r['weighted_f1']:.3f} & "
            f"{r['mean_latency_ms_per_image']:.2f} & {(h['total_training_time_sec'] / 60 if h else 0):.1f} \\\\")
    _write(gen, "results_table", "\n".join([
        r"\begin{table}[H]\centering\small",
        r"\caption{Held-out test results (identical test split for all models). Latency is mean forward-pass time per image "
        r"on the hardware in Section~\ref{sec:setup}; train time is wall-clock minutes.}\label{tab:results}",
        r"\begin{tabular}{lrrrrrrrr}\toprule",
        r"Model & Acc (\%) & Bal.\ acc (\%) & Macro P & Macro R & Macro F1 & W.\ F1 & ms/img & Train (min) \\\midrule",
        *(res_rows or [r"\multicolumn{9}{c}{\textit{[run evaluation]}}\\"]),
        r"\bottomrule\end{tabular}\end{table}",
    ]))

    # ---- per-class P/R/F1 table, all models side by side
    if done:
        classes = next(iter(done.values()))["class_names"]
        head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{_tex(PRETTY.get(m, m))}}}" for m in done)
        sub = " & ".join(["P & R & F1"] * len(done))
        rows = []
        for c in classes:
            cells = []
            for m, r in done.items():
                pc = r["per_class_report"][c]
                cells += [f"{pc['precision']:.2f}", f"{pc['recall']:.2f}", f"{pc['f1-score']:.2f}"]
            sup = int(next(iter(done.values()))["per_class_report"][c]["support"])
            rows.append(f"{_tex(c)} & {sup} & " + " & ".join(cells) + r" \\")
        _write(gen, "perclass_table", "\n".join([
            r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
            r"\caption{Per-class test precision (P), recall (R) and F1. $n$ = test images per class.}\label{tab:perclass}",
            r"\begin{tabular}{lr" + "ccc" * len(done) + r"}\toprule",
            rf" & & {head} \\",
            rf"Class & $n$ & {sub} \\\midrule",
            *rows,
            r"\bottomrule\end{tabular}\end{table}",
        ]))
        # top confusions per model
        lines = [r"\begin{table}[H]\centering\small",
                 r"\caption{Most frequent test confusions (true $\rightarrow$ predicted, count).}\label{tab:confusions}",
                 r"\begin{tabular}{lp{11cm}}\toprule", r"Model & Top confusions \\\midrule"]
        for m, r in done.items():
            items = ", ".join(f"{_tex(c['true_class'])}$\\rightarrow${_tex(c['pred_class'])} ({c['count']})" for c in r["top_confusions"][:5])
            lines.append(f"{_tex(PRETTY.get(m, m))} & {items or 'none'} \\\\")
        lines += [r"\bottomrule\end{tabular}\end{table}"]
        _write(gen, "confusions_table", "\n".join(lines))
    else:
        _write(gen, "perclass_table", r"\textit{[Per-class table: run evaluation.]}")
        _write(gen, "confusions_table", "")

    # ---- copy figures
    for f in config.figures_dir.glob("*.png"):
        shutil.copy2(f, figs / f.name)

    print(f"Report assets written to {gen} and {figs}")
    if banner:
        print("WARNING: results are synthetic or missing - the report will show a DRAFT banner.")
    return gen


if __name__ == "__main__":
    generate()
