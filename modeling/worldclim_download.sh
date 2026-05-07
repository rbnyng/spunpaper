#!/bin/bash

# Create a directory to store the data
data_dir="data"
mkdir -p "$data_dir"


# List of URLs
urls=(
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_tmin.zip"
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_tmax.zip"
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_tavg.zip"
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_prec.zip"
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_srad.zip"
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_wind.zip"
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_vapr.zip"
	"https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip"
	"https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_elev.zip"
)

# Download and unzip each file
for url in "${urls[@]}"; do
    filename=$(basename "$url")
    filepath="$data_dir/$filename"
    
    if [ -f "$filepath" ]; then
        echo "$filename exists skipping download."
    else
        echo "Downloading $filename..."
        curl -o "$filepath" "$url"
    fi
    
    echo "Unzipping $filename..."
    unzip -o "$filepath" -d "$data_dir"
    
    echo "Removing $filename..."
    rm "$filepath"
done

echo "Download complete"
