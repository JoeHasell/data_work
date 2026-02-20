"""
Create WID data with multiple income concepts and per capita adjustments.

Processes WID percentile data to create versions for:
- Pre-tax income (per adult and per capita)
- Post-tax income (per adult and per capita)

Output: modified/WID_percentiles_all_concepts.csv
"""

import pandas as pd
import os

def main():
    print("="*70)
    print("CREATING WID DATA WITH MULTIPLE INCOME CONCEPTS")
    print("="*70)

    # Combine individual country files
    print("\nCombining WID country files...")
    temp_dir = "inputs/temp_country_data"

    country_files = [f for f in os.listdir(temp_dir) if f.endswith('_percentiles.csv')]
    print(f"Found {len(country_files)} country files")

    if len(country_files) == 0:
        print("ERROR: No country files found!")
        return

    # Load and combine
    dfs = []
    for file in country_files:
        df = pd.read_csv(os.path.join(temp_dir, file))
        dfs.append(df)

    wid_percentiles = pd.concat(dfs, ignore_index=True)
    print(f"Combined into {len(wid_percentiles):,} rows")

    # Load population data
    print("\nLoading population data...")
    population_df = pd.read_csv('inputs/WID_aggregate_population.csv')
    print(f"Loaded {len(population_df):,} rows")

    # Load PPP data
    print("\nLoading PPP data...")
    ppp_df = pd.read_csv('inputs/WID_ppp.csv')
    print(f"Loaded {len(ppp_df):,} rows")

    # Filter to 2023
    year = 2023
    wid_percentiles = wid_percentiles[wid_percentiles['year'] == year].copy()
    population_df = population_df[population_df['year'] == year].copy()

    print(f"\nFiltered to {year}:")
    print(f"  Percentile data: {len(wid_percentiles):,} rows")
    print(f"  Population data: {len(population_df):,} rows")

    # Merge with population and PPP data
    wid_with_pop = wid_percentiles.merge(
        population_df[['country', 'adult_pop', 'total_pop']],
        on='country',
        how='left'
    ).merge(
        ppp_df[['country', 'ppp']],
        on='country',
        how='left'
    )

    print(f"\nAfter merge: {len(wid_with_pop):,} rows")
    print(f"Rows with missing population: {wid_with_pop[['adult_pop', 'total_pop']].isna().any(axis=1).sum()}")
    print(f"Rows with missing PPP: {wid_with_pop['ppp'].isna().sum()}")

    # Calculate per capita averages for both income concepts
    # NOTE: WID data comes in LOCAL CURRENCY - convert to PPP-adjusted international dollars
    # Then convert from annual to daily to match PIP
    # Pre-tax
    if 'avg_pretax' in wid_with_pop.columns:
        # Convert from local currency to PPP international dollars, then annual to daily
        wid_with_pop['avg_pretax_daily'] = (wid_with_pop['avg_pretax'] / wid_with_pop['ppp']) / 365

        # Calculate per capita (from per adult)
        wid_with_pop['avg_pretax_per_capita'] = (
            wid_with_pop['avg_pretax_daily'] * wid_with_pop['adult_pop']
        ) / wid_with_pop['total_pop']

        # Rename original for clarity (PPP-adjusted and converted to daily)
        wid_with_pop = wid_with_pop.rename(columns={'avg_pretax_daily': 'avg_pretax_per_adult'})
        wid_with_pop = wid_with_pop.rename(columns={'share_pretax': 'share_pretax'})

        # Drop the annual version
        wid_with_pop = wid_with_pop.drop(columns=['avg_pretax'])

        print(f"\n✓ Created pre-tax per capita averages (converted annual to daily)")

    # Post-tax
    if 'avg_posttax' in wid_with_pop.columns:
        # Convert from local currency to PPP international dollars, then annual to daily
        wid_with_pop['avg_posttax_daily'] = (wid_with_pop['avg_posttax'] / wid_with_pop['ppp']) / 365

        # Calculate per capita (from per adult)
        wid_with_pop['avg_posttax_per_capita'] = (
            wid_with_pop['avg_posttax_daily'] * wid_with_pop['adult_pop']
        ) / wid_with_pop['total_pop']

        # Rename original for clarity (PPP-adjusted and converted to daily)
        wid_with_pop = wid_with_pop.rename(columns={'avg_posttax_daily': 'avg_posttax_per_adult'})
        wid_with_pop = wid_with_pop.rename(columns={'share_posttax': 'share_posttax'})

        # Drop the annual version
        wid_with_pop = wid_with_pop.drop(columns=['avg_posttax'])

        print(f"✓ Created post-tax per capita averages (converted annual to daily)")

    # Calculate bin-specific populations from percentile ranges
    # Each percentile bin represents (p_high - p_low) fraction of the population
    wid_with_pop['bin_adult_pop'] = wid_with_pop['adult_pop'] * (wid_with_pop['p_high'] - wid_with_pop['p_low'])
    wid_with_pop['bin_total_pop'] = wid_with_pop['total_pop'] * (wid_with_pop['p_high'] - wid_with_pop['p_low'])

    print(f"\n✓ Calculated bin-specific populations from percentile ranges")

    # Select and reorder columns
    output_columns = [
        'country', 'percentile', 'year',
        'avg_pretax_per_adult', 'avg_pretax_per_capita', 'share_pretax',
        'avg_posttax_per_adult', 'avg_posttax_per_capita', 'share_posttax',
        'p_low', 'p_high', 'adult_pop', 'total_pop',
        'bin_adult_pop', 'bin_total_pop'
    ]

    # Only keep columns that exist
    output_columns = [col for col in output_columns if col in wid_with_pop.columns]
    wid_output = wid_with_pop[output_columns].copy()

    # Save to modified folder
    output_file = 'modified/WID_percentiles_all_concepts.csv'
    wid_output.to_csv(output_file, index=False)
    print(f"\n{output_file} saved with {len(wid_output):,} rows")

    # Display sample
    print("\nSample of data (first 5 rows):")
    print(wid_output.head())

    # Show statistics
    print("\nStatistics:")
    print(f"  Countries: {wid_output['country'].nunique()}")
    if 'avg_pretax_per_adult' in wid_output.columns:
        print(f"  Countries with pre-tax data: {wid_output['avg_pretax_per_adult'].notna().sum() // wid_output.groupby('country').size().iloc[0]}")
    if 'avg_posttax_per_adult' in wid_output.columns:
        print(f"  Countries with post-tax data: {wid_output['avg_posttax_per_adult'].notna().sum() // wid_output.groupby('country').size().iloc[0]}")

    print(f"  Adult population range: {wid_output['adult_pop'].min():,.0f} to {wid_output['adult_pop'].max():,.0f}")
    print(f"  Total population range: {wid_output['total_pop'].min():,.0f} to {wid_output['total_pop'].max():,.0f}")
    print(f"  Adult/Total ratio range: {(wid_output['adult_pop']/wid_output['total_pop']).min():.3f} to {(wid_output['adult_pop']/wid_output['total_pop']).max():.3f}")
    print(f"  Mean adult/total ratio: {(wid_output['adult_pop']/wid_output['total_pop']).mean():.3f}")

    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
