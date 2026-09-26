"""
Main CLI entrypoint for the Microcontroller & Component Classification project.
Allows running data preparation, training all 3 models, evaluation, and visualization.
"""

import argparse
import sys
from pathlib import Path
from mlproject.config import default_config
from mlproject.data.download_and_prep import prepare_dataset, generate_demo_dataset
from mlproject.train import train_model
from mlproject.evaluate import evaluate_model
from mlproject.visualize import plot_learning_curves, plot_model_comparison, plot_misclassified_samples


def run_pipeline():
    """Run full pipeline: prepare data, train 3 models, evaluate all, generate comparison figures."""
    print("=" * 70)
    print("RUNNING COMPLETE ITRI626 BENCHMARK PIPELINE")
    print("=" * 70)

    # 1. Prepare data
    print("\n--- STEP 1: PREPARING DATASET & SPLITS ---")
    prepare_dataset(config=default_config, generate_if_empty=True)

    # 2. Train 3 models from 3 distinct families
    models_to_train = [
        ("custom_cnn", 15, 1e-3),
        ("resnet18", 12, 3e-4),
        ("mobilenet_v3", 12, 3e-4),
    ]

    for model_name, epochs, lr in models_to_train:
        print(f"\n--- STEP 2: TRAINING MODEL: {model_name} ---")
        train_model(
            model_name=model_name,
            epochs=epochs,
            batch_size=32,
            lr=lr,
            config=default_config,
        )

        # Plot learning curves
        hist_file = default_config.models_dir / model_name / "history.json"
        if hist_file.exists():
            fig_path = plot_learning_curves(hist_file, config=default_config)
            print(f"Generated learning curves: {fig_path}")

    # 3. Evaluate on held-out test split
    print("\n--- STEP 3: EVALUATION ON HELD-OUT TEST SPLIT ---")
    for model_name, _, _ in models_to_train:
        evaluate_model(model_name=model_name, config=default_config)
        res_file = default_config.results_dir / f"{model_name}_results.json"
        if res_file.exists():
            plot_misclassified_samples(res_file, config=default_config)

    # 4. Generate comparison plots
    print("\n--- STEP 4: GENERATING MULTI-MODEL COMPARISON PLOTS ---")
    cmp_fig = plot_model_comparison(config=default_config)
    print(f"Comparative report figure generated: {cmp_fig}")
    print("\nBenchmark pipeline successfully completed!")


def main():
    parser = argparse.ArgumentParser(description="Microcontroller Image Classification CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Prepare command
    sub_prep = subparsers.add_parser("prepare", help="Prepare dataset and splits")
    sub_prep.add_argument("--demo", action="store_true", help="Generate demo dataset")
    sub_prep.add_argument("--samples", type=int, default=35, help="Samples per class for demo")

    # Train command
    sub_train = subparsers.add_parser("train", help="Train a model")
    sub_train.add_argument("--model", type=str, default="custom_cnn", choices=["custom_cnn", "resnet18", "resnet50", "mobilenet_v3", "vit_b_16"])
    sub_train.add_argument("--epochs", type=int, default=20)
    sub_train.add_argument("--batch-size", type=int, default=32)
    sub_train.add_argument("--lr", type=float, default=1e-3)
    sub_train.add_argument("--seed", type=int, default=42)

    # Evaluate command
    sub_eval = subparsers.add_parser("evaluate", help="Evaluate a model on test split")
    sub_eval.add_argument("--model", type=str, default="custom_cnn")

    # Compare command
    subparsers.add_parser("compare", help="Generate comparison charts across trained models")

    # Run-all pipeline
    subparsers.add_parser("run-all", help="Execute complete end-to-end benchmark")

    args = parser.parse_args()

    if args.command == "prepare":
        if args.demo:
            generate_demo_dataset(default_config.raw_data_dir, samples_per_class=args.samples)
        prepare_dataset(config=default_config)
    elif args.command == "train":
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            config=default_config,
        )
        hist = default_config.models_dir / args.model / "history.json"
        if hist.exists():
            plot_learning_curves(hist, config=default_config)
    elif args.command == "evaluate":
        evaluate_model(model_name=args.model, config=default_config)
        res = default_config.results_dir / f"{args.model}_results.json"
        if res.exists():
            plot_misclassified_samples(res, config=default_config)
    elif args.command == "compare":
        plot_model_comparison(config=default_config)
    elif args.command == "run-all":
        run_pipeline()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
