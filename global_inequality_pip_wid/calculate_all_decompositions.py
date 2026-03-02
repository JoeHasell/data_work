"""
Calculate all MLD decompositions and save results to CSV.

This script runs all inequality decomposition analyses and saves the results
to a single CSV file with verbose labels. Plotting scripts can then read from
this CSV to create visualizations.

Output: outputs/mld_decompositions.csv
"""

import pandas as pd
import numpy as np
from mld_decomposition import calculate_mld_decomposition


def load_pip_data(year=2023):
    """Load and aggregate PIP data."""
    print("\n" + "="*70)
    print("LOADING PIP DATA")
    print("="*70)

    url = "https://catalog.ourworldindata.org/garden/wb/2025-10-13/thousand_bins_distribution/thousand_bins_distribution.feather?nocache"
    df = pd.read_feather(url)
    df = df[df['year'] == year]

    # Aggregate to WID structure (101 bins)
    def assign_wid_bin(quantile_num):
        if quantile_num <= 990:
            percentile = (quantile_num - 1) // 10
            return f"p{percentile}p{percentile + 1}"
        elif quantile_num <= 999:
            return "p99p99.9"
        else:
            return "p99.9p100"

    df = df.copy()
    df['percentile'] = df['quantile'].apply(assign_wid_bin)

    agg_df = df.groupby(['country', 'year', 'percentile']).apply(
        lambda x: pd.Series({
            'pop': x['pop'].sum(),
            'average': np.average(x['avg'], weights=x['pop'])
        })
    ).reset_index()

    print(f"Loaded PIP data: {len(agg_df):,} rows, {agg_df['country'].nunique()} countries")
    return agg_df


def load_wid_data():
    """Load WID data with all income concepts."""
    print("\n" + "="*70)
    print("LOADING WID DATA")
    print("="*70)

    wid_df = pd.read_csv('modified/WID_percentiles_all_concepts.csv')
    print(f"Loaded WID data: {len(wid_df):,} rows, {wid_df['country'].nunique()} countries")
    return wid_df


def calculate_pip_decomposition(pip_df):
    """Calculate PIP (per capita) decomposition."""
    print("\n" + "="*70)
    print("CALCULATING: PIP (per capita)")
    print("="*70)

    results = calculate_mld_decomposition(pip_df, income_col='average', pop_col='pop')

    print(f"Global mean: ${results['global_mean']:.2f}/day")
    print(f"Countries: {results['num_countries']}")
    print(f"Total MLD: {results['total_mld']:.4f}")
    print(f"Between: {results['between_share']*100:.1f}%, Within: {results['within_share']*100:.1f}%")

    return results


def calculate_wid_decompositions(wid_df):
    """Calculate all WID decompositions."""
    decompositions = {}

    # WID Pre-tax (per adult)
    print("\n" + "="*70)
    print("CALCULATING: WID Pre-tax (per adult)")
    print("="*70)

    results = calculate_mld_decomposition(wid_df, income_col='avg_pretax_per_adult', pop_col='bin_adult_pop')
    print(f"Global mean: ${results['global_mean']:.2f}/day")
    print(f"Countries: {results['num_countries']}")
    print(f"Total MLD: {results['total_mld']:.4f}")
    print(f"Between: {results['between_share']*100:.1f}%, Within: {results['within_share']*100:.1f}%")
    decompositions['WID_pretax_per_adult'] = results

    # WID Pre-tax (per capita)
    print("\n" + "="*70)
    print("CALCULATING: WID Pre-tax (per capita)")
    print("="*70)

    results = calculate_mld_decomposition(wid_df, income_col='avg_pretax_per_capita', pop_col='bin_total_pop')
    print(f"Global mean: ${results['global_mean']:.2f}/day")
    print(f"Countries: {results['num_countries']}")
    print(f"Total MLD: {results['total_mld']:.4f}")
    print(f"Between: {results['between_share']*100:.1f}%, Within: {results['within_share']*100:.1f}%")
    decompositions['WID_pretax_per_capita'] = results

    # WID Post-tax (per capita)
    print("\n" + "="*70)
    print("CALCULATING: WID Post-tax (per capita)")
    print("="*70)

    results = calculate_mld_decomposition(wid_df, income_col='avg_posttax_per_capita', pop_col='bin_total_pop')
    print(f"Global mean: ${results['global_mean']:.2f}/day")
    print(f"Countries: {results['num_countries']}")
    print(f"Total MLD: {results['total_mld']:.4f}")
    print(f"Between: {results['between_share']*100:.1f}%, Within: {results['within_share']*100:.1f}%")
    decompositions['WID_posttax_per_capita'] = results

    return decompositions


def save_results_to_csv(pip_results, wid_results, output_file='outputs/mld_decompositions.csv'):
    """Save all decomposition results to a CSV with verbose labels."""

    rows = []

    # PIP row
    rows.append({
        'analysis_id': 'PIP_per_capita',
        'dataset': 'PIP',
        'income_concept': 'Consumption/Income',
        'population_basis': 'Per capita',
        'income_timing': 'Post-tax equivalent',
        'num_countries': pip_results['num_countries'],
        'total_population': pip_results['total_pop'],
        'global_mean_income_daily': pip_results['global_mean'],
        'total_mld': pip_results['total_mld'],
        'between_country_mld': pip_results['between_mld'],
        'within_country_mld': pip_results['within_mld'],
        'between_country_share': pip_results['between_share'],
        'within_country_share': pip_results['within_share']
    })

    # WID rows
    wid_configs = [
        ('WID_pretax_per_adult', 'WID', 'Pre-tax national income', 'Per adult', 'Pre-tax'),
        ('WID_pretax_per_capita', 'WID', 'Pre-tax national income', 'Per capita', 'Pre-tax'),
        ('WID_posttax_per_capita', 'WID', 'Post-tax disposable income', 'Per capita', 'Post-tax')
    ]

    for analysis_id, dataset, income_concept, pop_basis, income_timing in wid_configs:
        results = wid_results[analysis_id]
        rows.append({
            'analysis_id': analysis_id,
            'dataset': dataset,
            'income_concept': income_concept,
            'population_basis': pop_basis,
            'income_timing': income_timing,
            'num_countries': results['num_countries'],
            'total_population': results['total_pop'],
            'global_mean_income_daily': results['global_mean'],
            'total_mld': results['total_mld'],
            'between_country_mld': results['between_mld'],
            'within_country_mld': results['within_mld'],
            'between_country_share': results['between_share'],
            'within_country_share': results['within_share']
        })

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print("\n" + "="*70)
    print("RESULTS SAVED")
    print("="*70)
    print(f"\nSaved decomposition results to: {output_file}")
    print(f"Rows: {len(df)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Analysis ID':<30} {'Between':<12} {'Within':<12} {'Total MLD':<12}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['analysis_id']:<30} {row['between_country_share']*100:>6.1f}%      "
              f"{row['within_country_share']*100:>6.1f}%      {row['total_mld']:>8.4f}")

    return df


def main():
    print("="*70)
    print("CALCULATE ALL MLD DECOMPOSITIONS")
    print("="*70)
    print("\nThis script calculates all inequality decompositions and saves")
    print("results to CSV for use by plotting scripts.")

    # Load data
    pip_df = load_pip_data()
    wid_df = load_wid_data()

    # Calculate decompositions
    pip_results = calculate_pip_decomposition(pip_df)
    wid_results = calculate_wid_decompositions(wid_df)

    # Save to CSV
    results_df = save_results_to_csv(pip_results, wid_results)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  - Use plotting scripts to create visualizations from this CSV")
    print("  - See: plot_basic_decomposition.py, plot_income_concepts.py, etc.")
    print("="*70)


if __name__ == "__main__":
    main()
