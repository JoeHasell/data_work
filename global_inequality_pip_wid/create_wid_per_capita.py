"""
Create WID percentile data with per capita averages.

The original WID data has averages per adult. This script calculates
per capita averages by adjusting for the ratio of adults to total population.

Per capita average = (Per adult average * adult_pop) / total_pop
"""

import pandas as pd

def main():
    print("="*70)
    print("CREATING WID DATA WITH PER CAPITA AVERAGES")
    print("="*70)

    # Load WID percentile data
    print("\nLoading WID percentile data...")
    wid_percentiles = pd.read_csv('inputs/WID_percentiles.csv')
    print(f"Loaded {len(wid_percentiles):,} rows")

    # Load population data
    print("Loading population data...")
    population_df = pd.read_csv('inputs/WID_aggregate_population.csv')
    print(f"Loaded {len(population_df):,} rows")

    # Filter to 2023
    year = 2023
    wid_percentiles = wid_percentiles[wid_percentiles['year'] == year].copy()
    population_df = population_df[population_df['year'] == year].copy()

    print(f"\nFiltered to {year}:")
    print(f"  Percentile data: {len(wid_percentiles):,} rows")
    print(f"  Population data: {len(population_df):,} rows")

    # Merge with population data
    wid_with_pop = wid_percentiles.merge(
        population_df[['country', 'adult_pop', 'total_pop']],
        on='country',
        how='left'
    )

    print(f"\nAfter merge: {len(wid_with_pop):,} rows")
    print(f"Rows with missing population: {wid_with_pop[['adult_pop', 'total_pop']].isna().any(axis=1).sum()}")

    # Calculate per capita average
    # avg (per adult) * adult_pop / total_pop = avg (per capita)
    wid_with_pop['avg_per_capita'] = (wid_with_pop['avg'] * wid_with_pop['adult_pop']) / wid_with_pop['total_pop']

    # Rename original avg column for clarity
    wid_with_pop = wid_with_pop.rename(columns={'avg': 'avg_per_adult'})

    # Select and reorder columns
    output_columns = ['country', 'percentile', 'year', 'avg_per_adult', 'avg_per_capita',
                     'share', 'p_low', 'p_high', 'adult_pop', 'total_pop']
    wid_output = wid_with_pop[output_columns].copy()

    # Save to modified folder
    output_file = 'modified/WID_percentiles_with_per_capita.csv'
    wid_output.to_csv(output_file, index=False)
    print(f"\n{output_file} saved with {len(wid_output):,} rows")

    # Display sample
    print("\nSample of data (first 5 rows):")
    print(wid_output.head())

    # Show statistics
    print("\nStatistics:")
    print(f"  Countries: {wid_output['country'].nunique()}")
    print(f"  Adult population range: {wid_output['adult_pop'].min():,.0f} to {wid_output['adult_pop'].max():,.0f}")
    print(f"  Total population range: {wid_output['total_pop'].min():,.0f} to {wid_output['total_pop'].max():,.0f}")
    print(f"  Adult/Total ratio range: {(wid_output['adult_pop']/wid_output['total_pop']).min():.3f} to {(wid_output['adult_pop']/wid_output['total_pop']).max():.3f}")
    print(f"  Mean adult/total ratio: {(wid_output['adult_pop']/wid_output['total_pop']).mean():.3f}")

    # Show impact on averages
    ratio_impact = wid_output['avg_per_capita'] / wid_output['avg_per_adult']
    print(f"\nPer capita / Per adult average ratio:")
    print(f"  Min: {ratio_impact.min():.3f}")
    print(f"  Max: {ratio_impact.max():.3f}")
    print(f"  Mean: {ratio_impact.mean():.3f}")
    print(f"  Median: {ratio_impact.median():.3f}")

    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
