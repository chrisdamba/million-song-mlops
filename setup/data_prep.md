# Data Preparation

This document outlines the steps to prepare the Million Song Dataset for our ML pipeline using Parquet format.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Prerequisites](#prerequisites)
3. [Downloading the Dataset](#downloading-the-dataset)
4. [Data Extraction](#data-extraction)
5. [Data Cleaning](#data-cleaning)
6. [Feature Engineering](#feature-engineering)
7. [Data Storage](#data-storage)

## Dataset Overview

The Million Song Dataset subset consists of 10,000 songs, each with various audio features and metadata. Our goal is to prepare this data for machine learning tasks, focusing on predicting song popularity.

## Prerequisites

- Python 3.8+
- AWS CLI configured
- Required Python packages:
  - pandas
  - numpy
  - boto3
  - pyarrow
  - h5py (for initial data extraction)

Install the required packages:

```bash
pip install pandas numpy boto3 pyarrow h5py
```

## Downloading the Dataset

1. Download the Million Song Dataset subset:

```bash
wget http://static.echonest.com/millionsongsubset_full.tar.gz
```

2. Extract the downloaded file:

```bash
tar -xvzf millionsongsubset_full.tar.gz
```

## Data Extraction

Use the following script to extract data from the HDF5 files and save as Parquet:

```python
import os
import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def extract_song_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        song_data = {
            'song_id': f['metadata']['songs']['song_id'][0].decode('utf-8'),
            'title': f['metadata']['songs']['title'][0].decode('utf-8'),
            'artist_name': f['metadata']['songs']['artist_name'][0].decode('utf-8'),
            'duration': f['analysis']['songs']['duration'][0],
            'tempo': f['analysis']['songs']['tempo'][0],
            'loudness': f['analysis']['songs']['loudness'][0],
            'year': f['musicbrainz']['songs']['year'][0],
            'song_hotttnesss': f['metadata']['songs']['song_hotttnesss'][0]
        }
        
        # Extract timbre and chroma features
        segments_timbre = f['analysis']['segments_timbre'][:]
        segments_pitches = f['analysis']['segments_pitches'][:]
        
        song_data.update({
            f'timbre_{i}': segments_timbre[:, i].mean() for i in range(12)
        })
        song_data.update({
            f'chroma_{i}': segments_pitches[:, i].mean() for i in range(12)
        })
    
    return song_data

def process_dataset(root_dir):
    all_songs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                song_data = extract_song_data(file_path)
                all_songs.append(song_data)
    
    return pd.DataFrame(all_songs)

# Process the dataset
df = process_dataset('MillionSongSubset/data')

# Save to Parquet
table = pa.Table.from_pandas(df)
pq.write_table(table, 'million_song_subset.parquet')
```

Run this script to extract the data and save it as a Parquet file.

## Data Cleaning

Perform the following data cleaning steps:

```python
import pandas as pd
import pyarrow.parquet as pq

# Read the Parquet file
df = pq.read_table('million_song_subset.parquet').to_pandas()

# Handle missing values
numerical_cols = ['duration', 'tempo', 'loudness', 'year', 'song_hotttnesss']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

string_cols = ['title', 'artist_name']
df[string_cols] = df[string_cols].fillna('Unknown')

# Remove duplicates
df.drop_duplicates(subset='song_id', keep='first', inplace=True)

# Handle outliers
def remove_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df

for col in ['duration', 'tempo', 'loudness', 'song_hotttnesss']:
    df = remove_outliers(df, col)

# Save the cleaned data
table = pd.Table.from_pandas(df)
pq.write_table(table, 'million_song_subset_cleaned.parquet')
```

## Feature Engineering

Create new features that might be useful for predicting song popularity:

```python
import pandas as pd
import pyarrow.parquet as pq

# Read the cleaned Parquet file
df = pq.read_table('million_song_subset_cleaned.parquet').to_pandas()

# Create a 'decade' feature
df['decade'] = (df['year'] // 10) * 10

# Create a 'tempo_category' feature
df['tempo_category'] = pd.cut(df['tempo'], 
                              bins=[0, 60, 90, 120, 150, float('inf')],
                              labels=['Very Slow', 'Slow', 'Moderate', 'Fast', 'Very Fast'])

# Create a 'loudness_category' feature
df['loudness_category'] = pd.cut(df['loudness'], 
                                 bins=[-float('inf'), -20, -10, 0, float('inf')],
                                 labels=['Very Quiet', 'Quiet', 'Moderate', 'Loud'])

# Save the feature-engineered data
table = pa.Table.from_pandas(df)
pq.write_table(table, 'million_song_subset_featured.parquet')
```

## Data Storage

Store the prepared dataset in Amazon S3:

```python
import boto3
import pyarrow.parquet as pq

# Read the Parquet file
table = pq.read_table('million_song_subset_featured.parquet')

# Initialize S3 client
s3 = boto3.client('s3')

# Define bucket and file names
bucket_name = 'your-s3-bucket-name'
file_name = 'million_song_prepared.parquet'

# Upload file to S3
s3.put_object(Bucket=bucket_name, Key=file_name, Body=table.serialize().to_pybytes())

print(f"Data uploaded to S3://{bucket_name}/{file_name}")
```

Ensure you have the necessary permissions to write to the specified S3 bucket.

## Next Steps

After completing these data preparation steps, your dataset will be ready for model training. The Parquet file stored in S3 can be easily loaded and used in your ML pipeline. Proceed to the [Model Training](model_training.md) guide for the next steps in the ML pipeline.
```