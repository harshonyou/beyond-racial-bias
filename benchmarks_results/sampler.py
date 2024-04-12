import pandas as pd
import argparse
from collections import defaultdict
import random

def sample_skin_types(input_csv_path, output_csv_path, n):
    # Read the input CSV
    df = pd.read_csv(input_csv_path)

    # Ensure 'n' is an integer
    n = int(n)

    # Initialize a DataFrame to hold sampled data
    sampled_df_list = []

    # Filter and sample 'n' entries for each skin type
    for skin_type in df['skin_type'].unique():
        skin_type_df = df[df['skin_type'] == skin_type]

        # Ensure there are enough samples to sample 'n' entries
        if len(skin_type_df) < n:
            print(f"Not enough samples for skin type {skin_type}. Only {len(skin_type_df)} available.")
        else:
            sampled_df = skin_type_df.sample(n)
            sampled_df_list.append(sampled_df[['image_name', 'skin_type']])

    # Concatenate all sampled dataframes
    sampled_df = pd.concat(sampled_df_list)

    # Save the sampled entries to the output CSV file, only including 'image_name' and 'skin_type'
    sampled_df.to_csv(output_csv_path, columns=['image_name', 'skin_type'], index=False)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sample n entries for each skin type from a CSV file.")

    parser.add_argument('--input_csv_path', type=str, help="Path to the input CSV file.", default="./benchmarks_results/balanced_albedo_processed_images_log.csv")
    parser.add_argument('--output_csv_path', type=str, help="Path to save the output CSV file.", default="./benchmarks_results/sampled_output_csv.csv")
    parser.add_argument('--n', type=int, help="Number of entries to sample for each skin type.", default=10)

    args = parser.parse_args()

    # Call the function with the provided arguments
    sample_skin_types(args.input_csv_path, args.output_csv_path, args.n)