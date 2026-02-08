"""
Harmonize PIP and WID percentile data structures.

This script reads PIP data (with 1000 bins of 0.1% each) and aggregates it to match
the WID structure: 99 percentile buckets (P1-P99) + top 0.1% (P99.9-100) +
rest of top 1% (P99-P99.9).

Both datasets are combined with a unified column structure:
country, year, p_low, p_high, pop, average, source
"""

import pandas as pd
import numpy as np
import re


def parse_percentile_to_fractions(percentile_str):
    """
    Parse percentile string (e.g., 'p10p11', 'p99p99.9', 'p99.9p100')
    into p_low and p_high as fractions (0-1).

    Examples:
    - 'p10p11' -> (0.10, 0.11)
    - 'p99p99.9' -> (0.99, 0.999)
    - 'p99.9p100' -> (0.999, 1.0)
    """
    # Remove 'p' and split by remaining 'p'
    parts = percentile_str.replace('p', '', 1).split('p')
    p_low = float(parts[0]) / 100
    p_high = float(parts[1]) / 100
    return p_low, p_high


def load_pip_data(url, year=2023):
    """Load PIP percentile data from feather file and filter to specific year."""
    print("Loading PIP data...")
    df = pd.read_feather(url)
    print(f"Loaded {len(df):,} rows")

    # Filter to specific year
    df = df[df['year'] == year]
    print(f"Filtered to year {year}: {len(df):,} rows")
    return df


def aggregate_pip_to_wid_structure(pip_df):
    """
    Aggregate PIP data from 1000 bins (0.1% each) to WID structure.

    WID structure:
    - P0-P1, P1-P2, ..., P98-P99: 99 bins (1% each)
    - P99-P99.9: 1 bin (0.9% of population)
    - P99.9-P100: 1 bin (0.1% of population)

    Total: 101 bins
    """
    print("\nAggregating PIP data to WID structure...")

    # Create a mapping from PIP quantile (0.1% bins) to WID percentile bins
    def assign_wid_bin(quantile_num):
        """
        Assign each 0.1% quantile to appropriate WID bin.
        quantile_num ranges from 1-1000 in PIP data.
        """
        if quantile_num <= 990:
            # P0-P99: aggregate every 10 bins into 1% buckets
            # quantile 1-10 -> P0-P1
            # quantile 11-20 -> P1-P2, etc.
            percentile = (quantile_num - 1) // 10
            return f"p{percentile}p{percentile + 1}"
        elif quantile_num <= 999:
            # P99-P99.9 (quantiles 991-999, i.e., bins for 99.0-99.9%)
            return "p99p99.9"
        else:
            # P99.9-P100 (quantile 1000, i.e., bin for 99.9-100%)
            return "p99.9p100"

    # Assign WID bin labels
    pip_df = pip_df.copy()
    pip_df['percentile'] = pip_df['quantile'].apply(assign_wid_bin)

    # Group by country, year, and percentile bin
    # Sum population and calculate weighted average income
    agg_df = pip_df.groupby(['country', 'year', 'percentile']).apply(
        lambda x: pd.Series({
            'pop': x['pop'].sum(),
            'average': np.average(x['avg'], weights=x['pop'])
        })
    ).reset_index()

    # Parse percentile into p_low and p_high
    fractions = agg_df['percentile'].apply(parse_percentile_to_fractions)
    agg_df['p_low'] = fractions.apply(lambda x: x[0])
    agg_df['p_high'] = fractions.apply(lambda x: x[1])

    # Add source column
    agg_df['source'] = 'PIP'

    # Reorder columns
    agg_df = agg_df[['country', 'year', 'p_low', 'p_high', 'pop', 'average', 'source']]

    print(f"Aggregated to {len(agg_df):,} rows")
    print(f"Number of percentile bins: {agg_df.groupby(['country', 'year']).size().iloc[0] if len(agg_df) > 0 else 0}")

    return agg_df


def load_wid_data(year=2023):
    """Load and process WID data from local files."""
    print("\nLoading WID data from modified folder...")

    # Use the modified WID data with per capita averages
    wid_income_path = "modified/WID_percentiles_with_per_capita.csv"
    country_mapping_path = "inputs/country_mapping.csv"
    conversion_path = "inputs/WID_ppp.csv"

    # Load income data (now includes both per_adult and per_capita averages)
    income_df = pd.read_csv(wid_income_path)
    print(f"Loaded WID income data: {len(income_df):,} rows")

    # Load country mapping
    country_map_df = pd.read_csv(country_mapping_path)
    print(f"Loaded country mapping: {len(country_map_df):,} rows")

    # Load conversion factors
    conversion_df = pd.read_csv(conversion_path)
    print(f"Loaded conversion factors: {len(conversion_df):,} rows")

    return income_df, country_map_df, conversion_df


def process_wid_per_adult(income_df, country_map_df, conversion_df, year=2023):
    """
    Process WID data (per adult version) to match the harmonized structure.

    Returns DataFrame with columns: country, year, p_low, p_high, pop, average, source
    """
    print("\nProcessing WID data (per adult)...")

    # Filter to specific year
    income_df = income_df[income_df['year'] == year].copy()
    conversion_df = conversion_df[conversion_df['year'] == year].copy()

    print(f"Filtered income data to {year}: {len(income_df):,} rows")
    print(f"Filtered conversion data to {year}: {len(conversion_df):,} rows")

    # Map country codes to PIP country names
    country_map_dict = dict(zip(country_map_df['country'], country_map_df['PIP country name']))
    income_df['country_name'] = income_df['country'].map(country_map_dict)

    # Remove rows where country name is missing (not in PIP)
    income_df = income_df.dropna(subset=['country_name'])
    print(f"After country mapping: {len(income_df):,} rows")

    # Calculate population for each percentile bin using ADULT population
    # pop = adult_pop * (p_high - p_low)
    income_df['pop'] = income_df['adult_pop'] * (income_df['p_high'] - income_df['p_low'])

    # Merge with conversion factors (PPP)
    conversion_overall = conversion_df[conversion_df['percentile'] == 'p0p100'][['country', 'ppp']]
    income_df = income_df.merge(
        conversion_overall,
        on='country',
        how='left'
    )

    # Convert local currency to international dollars and annual to daily
    # Use avg_per_adult column
    income_df['average'] = income_df['avg_per_adult'] / income_df['ppp'] / 365

    # Add source column
    income_df['source'] = 'WID_per_adult'

    # Select and rename columns to match harmonized structure
    result_df = income_df[['country_name', 'year', 'p_low', 'p_high', 'pop', 'average', 'source']].copy()
    result_df = result_df.rename(columns={'country_name': 'country'})

    # Remove any rows with missing values
    result_df = result_df.dropna()

    print(f"Final WID (per adult) data: {len(result_df):,} rows")
    print(f"Countries: {result_df['country'].nunique()}")

    return result_df


def process_wid_per_capita(income_df, country_map_df, conversion_df, year=2023):
    """
    Process WID data (per capita version) to match the harmonized structure.

    Returns DataFrame with columns: country, year, p_low, p_high, pop, average, source
    """
    print("\nProcessing WID data (per capita)...")

    # Filter to specific year
    income_df = income_df[income_df['year'] == year].copy()
    conversion_df = conversion_df[conversion_df['year'] == year].copy()

    # Map country codes to PIP country names
    country_map_dict = dict(zip(country_map_df['country'], country_map_df['PIP country name']))
    income_df['country_name'] = income_df['country'].map(country_map_dict)

    # Remove rows where country name is missing (not in PIP)
    income_df = income_df.dropna(subset=['country_name'])

    # Calculate population for each percentile bin using TOTAL population
    # pop = total_pop * (p_high - p_low)
    income_df['pop'] = income_df['total_pop'] * (income_df['p_high'] - income_df['p_low'])

    # Merge with conversion factors (PPP)
    conversion_overall = conversion_df[conversion_df['percentile'] == 'p0p100'][['country', 'ppp']]
    income_df = income_df.merge(
        conversion_overall,
        on='country',
        how='left'
    )

    # Convert local currency to international dollars and annual to daily
    # Use avg_per_capita column
    income_df['average'] = income_df['avg_per_capita'] / income_df['ppp'] / 365

    # Add source column
    income_df['source'] = 'WID_per_capita'

    # Select and rename columns to match harmonized structure
    result_df = income_df[['country_name', 'year', 'p_low', 'p_high', 'pop', 'average', 'source']].copy()
    result_df = result_df.rename(columns={'country_name': 'country'})

    # Remove any rows with missing values
    result_df = result_df.dropna()

    print(f"Final WID (per capita) data: {len(result_df):,} rows")
    print(f"Countries: {result_df['country'].nunique()}")

    return result_df


def main():
    """Main execution function."""
    year = 2023

    # PIP data URL
    pip_url = "https://catalog.ourworldindata.org/garden/wb/2025-10-13/thousand_bins_distribution/thousand_bins_distribution.feather?nocache"

    # Load and process PIP data
    pip_df = load_pip_data(pip_url, year=year)
    pip_harmonized = aggregate_pip_to_wid_structure(pip_df)

    print("\nSample of harmonized PIP data:")
    print(pip_harmonized.head(10))

    # Load WID data
    wid_income, wid_country_map, wid_conversion = load_wid_data(year=year)

    # Process WID data - both per adult and per capita versions
    wid_per_adult = process_wid_per_adult(wid_income, wid_country_map, wid_conversion, year=year)
    wid_per_capita = process_wid_per_capita(wid_income, wid_country_map, wid_conversion, year=year)

    # Combine all three datasets
    harmonized_df = pd.concat([pip_harmonized, wid_per_adult, wid_per_capita], ignore_index=True)

    print(f"\n" + "="*70)
    print("COMBINED DATASET SUMMARY")
    print("="*70)
    print(f"Total rows: {len(harmonized_df):,}")
    print(f"  PIP rows: {len(pip_harmonized):,}")
    print(f"  WID (per adult) rows: {len(wid_per_adult):,}")
    print(f"  WID (per capita) rows: {len(wid_per_capita):,}")

    # Save harmonized data
    output_file = "modified/pip_wid_harmonized.csv"
    harmonized_df.to_csv(output_file, index=False)
    print(f"\nHarmonized data saved to: {output_file}")

    # Display sample of final output
    print("\nSample of final harmonized data:")
    print(harmonized_df.head(20))

    # Show statistics by source
    print("\n" + "="*70)
    print("DATA BY SOURCE")
    print("="*70)
    source_stats = harmonized_df.groupby('source').agg({
        'country': 'nunique',
        'pop': 'sum'
    }).rename(columns={'country': 'num_countries', 'pop': 'total_population'})
    print(source_stats)

    return harmonized_df


if __name__ == "__main__":
    harmonized_df = main()
