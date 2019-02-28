#!/usr/bin/env bash

# Can execute script from anywhere
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ..

mv datasets/yelp_dataset/yelp_academic_dataset_review.json datasets/yelp_dataset/review.json
PYTHONPATH=. python data_loaders/yelp_dataset.py