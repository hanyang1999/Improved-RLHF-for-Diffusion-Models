ulimit -n 64000; accelerate launch scripts/train_continuous.py --config config/dgx_score.py:incompressibility 
wait
accelerate launch scripts/train_continuous.py --config config/dgx_noise.py:incompressibility