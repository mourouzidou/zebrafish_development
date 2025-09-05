python3 scripts/train.py \
--split_dir data/embryo/training/train_atac_L2k_g11_embryo_cpm_l0_q0__seqs/80_q0__seqs/80_20_chrom_split \
--dataset embryo_cpm_l0_q0 \ --loss_name poisson_log \
--model_name dilated_baseline_model \
--batch_size 64 \
--lr 1e-3 --weight_decay 1e-4 \
--use_test_as_val \
--num_epochs 300 \
--early_stopping_patience 10