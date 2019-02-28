#!/usr/bin/env bash

# Can execute script from anywhere
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

pip install -r requirements.txt

python -m nltk.downloader punkt
python -m nltk.downloader stopwords