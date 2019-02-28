#!/usr/bin/env bash

# Can execute script from anywhere
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ..

mkdir -p outputs/eval/yelp/n_docs_8
mkdir -p outputs/eval/amazon/n_docs_8

mkdir -p datasets/yelp_dataset/processed
mkdir -p datasets/amazon_dataset/processed

# Experiments will go here
mkdir -p checkpoints/lm/mlstm/yelp
mkdir -p checkpoints/clf/cnn/yelp
mkdir -p checkpoints/sum/mlstm/yelp

mkdir -p checkpoints/lm/mlstm/amazon
mkdir -p checkpoints/clf/cnn/amazon
mkdir -p checkpoints/sum/mlstm/amazon

# Pretrained / trained models will go here
mkdir -p stable_checkpoints/lm/mlstm/yelp
mkdir -p stable_checkpoints/clf/cnn/yelp
mkdir -p stable_checkpoints/sum/mlstm/yelp

mkdir -p stable_checkpoints/lm/mlstm/amazon
mkdir -p stable_checkpoints/clf/cnn/amazon
mkdir -p stable_checkpoints/sum/mlstm/amazon
