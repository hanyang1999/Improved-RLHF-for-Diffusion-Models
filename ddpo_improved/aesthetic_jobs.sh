ulimit -n 64000; accelerate launch scripts/train_continuous.py --config config/dgx_noise.py:aesthetic
wait
accelerate launch scripts/train_continuous.py --config config/dgx_score.py:aesthetic