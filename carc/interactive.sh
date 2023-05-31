#!/bin/bash

salloc \
--account=shrikann_35 \
--time=2-00:00:00 \
--nodes=1 \
--ntasks=1 \
--cpus-per-task=16 \
--mem=64G \
--partition=gpu \
--gres=gpu:a40:1 \