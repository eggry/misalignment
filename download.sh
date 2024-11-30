#!/bin/bash

# MIT License

# Copyright (c) 2024 Yichen Gong, Delong Ran

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -e

echo "Start to download HarmfulSafeRLHF datasets..."

cd data/training/exploitingGPT4api &&

curl --silent -L \
    "https://github.com/AlignmentResearch/gpt-4-novel-apis-attacks/raw/c5b7149865ec4d2eea1cd05b5f1b43f3b24d9859/datasets/HarmfulSafeRLHF-10.jsonl" \
    -o "HarmfulSafeRLHF-10.jsonl" &&
curl --silent -L \
    "https://github.com/AlignmentResearch/gpt-4-novel-apis-attacks/raw/c5b7149865ec4d2eea1cd05b5f1b43f3b24d9859/datasets/HarmfulSafeRLHF-100.jsonl" \
    -o "HarmfulSafeRLHF-100.jsonl" &&
sha256sum -c files.sha256sum \
    && echo "Download HarmfulSafeRLHF datasets successful." \
    || echo "Failed to download HarmfulSafeRLHF datasets!"

cd ~-

echo "Start to download shadow-alignment datasets..."

cd data/training/SA &&

curl --silent -L \
    "https://huggingface.co/datasets/CherryDurian/shadow-alignment/resolve/1dcf76aca28c2b0886ea96712d08f8e365234977/data/eval-00000-of-00001-46dfb353534cb3f5.parquet" \
    -o "eval-00000-of-00001-46dfb353534cb3f5.parquet" &&
curl --silent -L \
    "https://huggingface.co/datasets/CherryDurian/shadow-alignment/resolve/1dcf76aca28c2b0886ea96712d08f8e365234977/data/train-00000-of-00001-980e7d9e9ef05341.parquet" \
    -o "train-00000-of-00001-980e7d9e9ef05341.parquet" &&
sha256sum -c files.sha256sum \
    && echo "Download shadow-alignment datasets successful." \
    || echo "Failed to download shadow-alignment datasets!"

cd ~-

echo "Start to download StrongReject datasets..."

cd data/evaluation/strongreject &&

curl --silent -L \
    "https://github.com/alexandrasouly/strongreject/raw/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/previous_versions/2024_02/strongreject_dataset.csv" \
    -o "strongreject_dataset.csv" &&
curl --silent -L \
    "https://github.com/alexandrasouly/strongreject/raw/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/previous_versions/2024_02/strongreject_small_dataset.csv" \
    -o "strongreject_small_dataset.csv" &&
sha256sum -c files.sha256sum \
    && echo "Download StrongReject datasets successful." \
    || echo "Failed to download StrongReject datasets!"

cd ~-
