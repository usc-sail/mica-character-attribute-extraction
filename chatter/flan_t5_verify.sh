#!/bin/bash

for i in {0..13}; do
    sbatch flan_t5_batch_verify.sh $i
done