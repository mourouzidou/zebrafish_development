import os, sys, argparse
from pathlib import Path
from zebra_dev.utils import training

THIS = Path(__file__).resolve()
REPO = THIS.parents[1]              # zebrafish_development/ == repo root
SRC  = REPO / "src"
sys.path.insert(0, str(SRC))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--dataset",   required=True)
    ap.add_argument("--model_name", default="dilated_baseline_model")
    ap.add_argument("--loss_name",  default="mse", choices=["mse","poisson_log"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--use_test_as_val", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_epochs", type=int, default=300)
    ap.add_argument("--early_stopping_patience", type=int, default=10)
    ap.add_argument("--cuda", type=str, default="")
    ap.add_argument("--out_root", default="runs")  # repo_root/runs relative by default
    args = ap.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = (REPO / split_dir).resolve()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO / out_root).resolve()

    training.train_on_split(
        split_dir=split_dir,
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
        out_root=out_root,   
    )

if __name__ == "__main__":
    main()
