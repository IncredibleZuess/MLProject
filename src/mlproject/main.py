"""
Command-line entry point:  python -m mlproject <command>

  prepare        clean data/raw, build leakage-free splits (+ --demo for synthetic test data)
  train          train one model            (--model custom_cnn|resnet18|vit_b_16|mobilenet_v3)
  evaluate       evaluate one model on the held-out test split
  compare        comparison figures across all trained models
  report-assets  regenerate LaTeX tables + figures used by report/report.tex
  run-all        prepare -> train 3 models -> evaluate -> figures -> report assets
"""

import argparse

from mlproject.config import default_config
from mlproject.data.download_and_prep import generate_demo_dataset, prepare_dataset
from mlproject.evaluate import evaluate_model
from mlproject.report_assets import generate as generate_report_assets
from mlproject.train import train_model
from mlproject.visualize import (
    plot_class_distribution,
    plot_class_examples,
    plot_correct_samples,
    plot_learning_curves,
    plot_misclassified_samples,
    plot_model_comparison,
    plot_per_class_f1,
    plot_validation_comparison,
)

cfg = default_config
cfg.ensure_directories()


def dataset_figures() -> None:
    print("Dataset figures:", plot_class_distribution(config=cfg), plot_class_examples(config=cfg))


def post_train(model_name: str) -> None:
    hist = cfg.models_dir / model_name / "history.json"
    if hist.exists():
        plot_learning_curves(hist, config=cfg)


def post_eval(model_name: str) -> None:
    res = cfg.results_dir / f"{model_name}_results.json"
    if res.exists():
        plot_misclassified_samples(res, config=cfg)
        plot_correct_samples(res, config=cfg)


def compare(models) -> None:
    plot_model_comparison(config=cfg)
    plot_validation_comparison(models, config=cfg)
    plot_per_class_f1(models, config=cfg)


def run_pipeline(models=None, demo: bool = False, quick_epochs: int = None) -> None:
    models = models or cfg.model_families
    print("=" * 70 + "\nITRI626 FULL PIPELINE: " + ", ".join(models) + "\n" + "=" * 70)
    prepare_dataset(config=cfg, generate_if_empty=demo)
    dataset_figures()
    for m in models:
        train_model(m, epochs=quick_epochs, config=cfg)
        post_train(m)
    for m in models:
        evaluate_model(m, config=cfg)
        post_eval(m)
    compare(models)
    generate_report_assets(models, config=cfg)
    print("\nPipeline complete. Re-compile report/report.tex to update the report.")


def main() -> None:
    p = argparse.ArgumentParser(description="ITRI626 microcontroller & component classification")
    sub = p.add_subparsers(dest="command")
    models = list(cfg.model_settings)

    sp = sub.add_parser("prepare", help="Clean data and create splits")
    sp.add_argument("--demo", action="store_true", help="Generate SYNTHETIC demo images first (pipeline test only)")
    sp.add_argument("--samples", type=int, default=35)
    sp.add_argument("--no-group-split", action="store_true", help="Plain stratified split (not recommended)")

    st = sub.add_parser("train", help="Train a model (defaults from config.model_settings)")
    st.add_argument("--model", default="custom_cnn", choices=models)
    st.add_argument("--epochs", type=int)
    st.add_argument("--batch-size", type=int)
    st.add_argument("--accum-steps", type=int)
    st.add_argument("--lr", type=float)
    st.add_argument("--seed", type=int)

    se = sub.add_parser("evaluate", help="Evaluate a model on the test split")
    se.add_argument("--model", default="custom_cnn", choices=models)

    sub.add_parser("compare", help="Comparison figures across trained models")
    sub.add_parser("report-assets", help="Regenerate report tables and figures")

    sr = sub.add_parser("run-all", help="Run the complete pipeline")
    sr.add_argument("--demo", action="store_true", help="Use synthetic data if data/raw is empty")
    sr.add_argument("--quick", type=int, default=None, metavar="EPOCHS", help="Override epochs (smoke test)")
    sr.add_argument("--models", nargs="+", choices=models, default=None)

    a = p.parse_args()
    if a.command == "prepare":
        if a.demo:
            generate_demo_dataset(cfg.raw_data_dir, samples_per_class=a.samples)
        prepare_dataset(config=cfg, group_aware=not a.no_group_split)
        dataset_figures()
    elif a.command == "train":
        train_model(a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, accum_steps=a.accum_steps,
                    seed=a.seed, config=cfg)
        post_train(a.model)
    elif a.command == "evaluate":
        evaluate_model(a.model, config=cfg)
        post_eval(a.model)
    elif a.command == "compare":
        compare(cfg.model_families)
    elif a.command == "report-assets":
        generate_report_assets(config=cfg)
    elif a.command == "run-all":
        run_pipeline(models=a.models, demo=a.demo, quick_epochs=a.quick)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
