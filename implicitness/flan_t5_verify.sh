#!/bin/bash

for i in {0..15}; do
    sbatch flan_t5_batch_verify.sh $i 16
done