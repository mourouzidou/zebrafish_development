import os
import argparse
from zebra_dev.utils.training import train_on_split

def main():
    ap = argparse.ArgumentParser(description="Train a model from a prepared split folder")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", default="dilated_baseline_model")
    ap.add_argument("--loss_name", default="mse", choices=["mse", "poisson_log"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--use_test_as_val", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_epochs", type=int, default=300)
    ap.add_argument("--early_stopping_patience", type=int, default=10)
    ap.add_argument("--cuda", type=str, default="", help="e.g. '0' or '1'")
    args = ap.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    train_on_split(
        split_dir=args.split_dir,
        dataset=args.dataset,
        model_name=args.model_name,
        loss_name=args.loss_name,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_test_as_val=args.use_test_as_val,
        val_frac=args.val_frac,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )