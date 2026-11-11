"""Command-line entry point: python train.py [--n-iter N] [--cv K]"""
import argparse
import sys

from src import config

logger = config.setup_logging()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train ChurnSense models.")
    parser.add_argument("--n-iter", type=int, default=config.N_ITER_SEARCH,
                        help="RandomizedSearchCV iterations per model")
    parser.add_argument("--cv", type=int, default=config.CV_FOLDS, help="CV folds")
    parser.add_argument("--data", type=str, default=None, help="Path to the churn CSV")
    args = parser.parse_args(argv)

    from src.training import run_training

    try:
        out = run_training(data_path=args.data, n_iter=args.n_iter, cv_folds=args.cv)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1

    meta = out["metadata"]
    m = meta["test_metrics"]
    print("\n" + "=" * 60)
    print(f"Selected model : {meta['selected_algorithm']}  ({meta['model_version']})")
    print(f"CV ROC-AUC     : {meta['cv_roc_auc']:.4f}")
    print(f"Test accuracy  : {m['accuracy']:.4f}")
    print(f"Test precision : {m['precision']:.4f}")
    print(f"Test recall    : {m['recall']:.4f}")
    print(f"Test F1        : {m['f1']:.4f}")
    print(f"Test ROC-AUC   : {m['roc_auc']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
