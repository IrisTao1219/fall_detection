#!/usr/bin/env bash

set -u

BASE_URL="https://fenix.ur.edu.pl/~mkepski/ds/data"
OUTPUT_ROOT="raw/adl"

mkdir -p "${OUTPUT_ROOT}"

for i in $(seq -w 1 40); do
    filename="adl-${i}-cam0-rgb.zip"
    url="${BASE_URL}/${filename}"

    echo "Downloading ${filename}..."
    curl --fail --location --continue-at - \
        --retry 5 \
        --retry-delay 5 \
        --connect-timeout 30 \
        --output "/raw/${filename}" \
        "${url}"

    if [ $? -ne 0 ]; then
        echo "Download failed: ${filename}" >&2
        exit 1
    fi

      
    output_dir="${OUTPUT_ROOT}/${filename%.zip}"
    mkdir -p "${output_dir}"
    echo "Extracting ${filename} -> ${output_dir}/..."
    unzip -q -o "raw/${filename}" -d "${output_dir}"

    if [ $? -ne 0 ]; then
        echo "Extraction failed: ${filename}" >&2
        exit 1
    fi

    
    csv_filename="adl-${i}-data.csv"
    csv_url="${BASE_URL}/${csv_filename}"

    echo "Downloading ${csv_filename}..."
    curl --fail --location --continue-at - \
        --retry 5 \
        --retry-delay 5 \
        --connect-timeout 30 \
        --output "${OUTPUT_ROOT}/${csv_filename}" \
        "${csv_url}"

    if [ $? -ne 0 ]; then
        echo "Download failed: ${csv_filename}" >&2
        exit 1
    fi
done

echo "All adl RGB archives and synchronization CSV files (01-40) downloaded successfully."
