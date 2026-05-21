#!/bin/bash
# Downloads external datasets needed for this project.
# Safe to re-run — skips files that already exist.

set -e
cd "$(dirname "$0")/.."

mkdir -p Data/bryant
cd Data/bryant

if [ ! -f allseqs_20191230.csv ]; then
    echo "Downloading Bryant et al. AAV2 dataset..."
    curl -L -o allseqs.csv.zip \
        https://github.com/alibashir/aav/raw/main/allseqs_20191230.csv.zip
    unzip allseqs.csv.zip
    rm allseqs.csv.zip
else
    echo "Bryant main CSV already present, skipping."
fi

if [ ! -f ValidationChipwithModelScores.csv ]; then
    echo "Downloading Bryant validation chip with model scores..."
    curl -L -o vchip.csv.bz2 \
        https://github.com/alibashir/aav/raw/main/ValidationChipwithModelScores.csv.bz2
    bzip2 -d vchip.csv.bz2
    mv vchip.csv ValidationChipwithModelScores.csv
else
    echo "Validation chip CSV already present, skipping."
fi

echo "Done. Files in Data/bryant/:"
ls -lh