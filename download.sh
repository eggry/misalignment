
echo "Start to download HarmfulSafeRLHF datasets..."

pushd data/training/exploitingGPT4api &&

wget "https://github.com/AlignmentResearch/gpt-4-novel-apis-attacks/raw/c5b7149865ec4d2eea1cd05b5f1b43f3b24d9859/datasets/HarmfulSafeRLHF-10.jsonl" &&
wget "https://github.com/AlignmentResearch/gpt-4-novel-apis-attacks/raw/c5b7149865ec4d2eea1cd05b5f1b43f3b24d9859/datasets/HarmfulSafeRLHF-100.jsonl" &&
sha256sum -c files.sha256sum \
    && echo "Download HarmfulSafeRLHF datasets successful." \
    || echo "Failed to download HarmfulSafeRLHF datasets!" 

popd

echo "Start to download shadow-alignment datasets..."

pushd data/training/SA &&

wget "https://huggingface.co/datasets/CherryDurian/shadow-alignment/resolve/1dcf76aca28c2b0886ea96712d08f8e365234977/data/eval-00000-of-00001-46dfb353534cb3f5.parquet" &&
wget "https://huggingface.co/datasets/CherryDurian/shadow-alignment/resolve/1dcf76aca28c2b0886ea96712d08f8e365234977/data/train-00000-of-00001-980e7d9e9ef05341.parquet" &&
sha256sum -c files.sha256sum \
    && echo "Download shadow-alignment datasets successful." \
    || echo "Failed to download shadow-alignment datasets!" 

popd

echo "Start to download StrongReject datasets..."

pushd data/evaluation/strongreject &&

wget "https://github.com/alexandrasouly/strongreject/raw/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/previous_versions/2024_02/strongreject_dataset.csv" &&
wget "https://github.com/alexandrasouly/strongreject/raw/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/previous_versions/2024_02/strongreject_small_dataset.csv" &&
sha256sum -c files.sha256sum \
    && echo "Download StrongReject datasets successful." \
    || echo "Failed to download StrongReject datasets!" 

popd
