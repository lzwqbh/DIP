# DIP


## About

This is the source code for paper _Teach Structure Features to Cooperate with Node Embeddings_ in IJCNN 2025.

The source code is based on [WALKPOOL](https://github.com/DaDaCheng/WalkPooling/), [ELPH](https://github.com/melifluos/subgraph-sketching), and [SIEG](https://github.com/anonymous20221001/SIEG_OGB).

## Run

### Quick start

For evaluating DIP-enhanced WalkPool:

	cd ./DIP_WP
    python ./src/eva_wp.py --init-representation gae --gpu-num 0 --data-nam cora --test-r 0.45 --stru-e 1
For evaluating DIP-enhanced ELPH:

	cd ./DIP_ELPH/src
    python runners/run.py --dataset-n cora --stru_enh 1 --sco_enh 1 --reps 5 --test_pct 0.45 --gpu-n 0
For evaluating DIP-enhanced SIEG:

	cd ./DIP_SIEG
    python train.py --device 0 --dataset cora --seed 1 --test-ratio 0.45 --dot_enh 1


## Reference


If you find our work useful in your research, please cite our paper in the future:



