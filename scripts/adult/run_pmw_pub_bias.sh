#!/bin/bash

DATASET=adult
ADULT_SEED=0

NUM_RUNS=5
MARGINAL=3
WORKLOAD=256
WORKLOAD_SEED=0
FRAC=0.1
FRAC_SEED=0
BIAS_ATTR=sex

for PERTURB in 0 -0.05 -0.1 -0.2 0.05 0.1 0.2 0.3 0.45 0.65
do
  for EPSILON in 1.0 0.5 0.25 0.2 0.15 0.1
  do
    for T in 5 10 25 50 75 100 150 200
    do
      python pmw_pub_bias.py --dataset $DATASET --adult_seed $ADULT_SEED \
      --num_runs $NUM_RUNS --marginal $MARGINAL \
      --workload $WORKLOAD --workload_seed $WORKLOAD_SEED \
      --pub_frac $FRAC --frac_seed $FRAC_SEED \
      --bias_attr $BIAS_ATTR --perturb $PERTURB \
      --epsilon $EPSILON --T $T --permute
    done
  done
done