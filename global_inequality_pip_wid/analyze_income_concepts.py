"""
Compare global inequality decomposition across income concepts.

Compares four data sources:
1. PIP (per capita, pre-tax equivalent)
2. WID pre-tax (per adult)
3. WID pre-tax (per capita)
4. WID post-tax (per capita)

Creates MLD decomposition analysis and bar chart visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def calculate_mld_decomposition(df, avg_col, pop_col):
    """
    Calculate Mean Log Deviation (MLD) decomposition.

    Args:
        df: DataFrame with percentile data
        avg_col: Column name for average income
        pop_col: Column name for population

    Returns:
        dict with total_mld, between_mld, within_mld, between_share, within_share
    """
    # Filter out rows with missing data
    df = df[df[avg_col].notna() & (df[avg_col] > 0)].copy()

    # Calculate global mean
    total_income = (df[avg_col] * df[pop_col]).sum()
    total_pop = df[pop_col].sum()
    global_mean = total_income / total_pop

    # Calculate country means
    country_stats = df.groupby('country').apply(
        lambda x: pd.Series({
            'country_mean': (x[avg_col] * x[pop_col]).sum() / x[pop_col].sum(),
            'country_pop': x[pop_col].sum()
        })
    ).reset_index()

    # Between-country MLD
    between_mld = ((country_stats['country_pop'] / total_pop) *
                   np.log(global_mean / country_stats['country_mean'])).sum()

    # Within-country MLD for each country
    within_mlds = []
    for country in df['country'].unique():
        country_df = df[df['country'] == country].copy()
        country_mean = (country_df[avg_col] * country_df[pop_col]).sum() / country_df[pop_col].sum()
        country_pop = country_df[pop_col].sum()

        # MLD within this country
        country_mld = ((country_df[pop_col] / country_pop) *
                      np.log(country_mean / country_df[avg_col])).sum()

        within_mlds.append({
            'country': country,
            'within_mld': country_mld,
            'pop_weight': country_pop / total_pop
        })

    within_df = pd.DataFrame(within_mlds)
    within_mld = (within_df['within_mld'] * within_df['pop_weight']).sum()

    # Total MLD
    total_mld = between_mld + within_mld

    return {
        'total_mld': total_mld,
        'between_mld': between_mld,
        'within_mld': within_mld,
        'between_share': between_mld / total_mld if total_mld > 0 else 0,
        'within_share': within_mld / total_mld if total_mld > 0 else 0,
        'global_mean': global_mean,
        'total_pop': total_pop,
        'num_countries': df['country'].nunique()
    }


def load_pip_data(year=2023):
    """Load PIP data for comparison."""
    print("\nLoading PIP data...")
    url = "https://catalog.ourworldindata.org/garden/wb/2025-10-13/thousand_bins_distribution/thousand_bins_distribution.feather?nocache"

    df = pd.read_feather(url)
    df = df[df['year'] == year]

    # Aggregate to match WID structure (101 bins)
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

    # Group by country and percentile
    agg_df = df.groupby(['country', 'year', 'percentile']).apply(
        lambda x: pd.Series({
            'pop': x['pop'].sum(),
            'average': np.average(x['avg'], weights=x['pop'])
        })
    ).reset_index()

    # Parse percentile
    def parse_percentile(perc_str):
        parts = perc_str.replace('p', '', 1).split('p')
        return float(parts[0]) / 100, float(parts[1]) / 100

    fractions = agg_df['percentile'].apply(parse_percentile)
    agg_df['p_low'] = fractions.apply(lambda x: x[0])
    agg_df['p_high'] = fractions.apply(lambda x: x[1])

    print(f"Loaded PIP data: {len(agg_df):,} rows, {agg_df['country'].nunique()} countries")

    return agg_df


def main():
    print("="*70)
    print("INCOME CONCEPT COMPARISON: GLOBAL INEQUALITY DECOMPOSITION")
    print("="*70)

    year = 2023

    # Load PIP data (download if not cached)
    print("\nFetching PIP data from URL...")
    pip_df = load_pip_data(year=year)

    # Load WID data with all concepts
    print("\nLoading WID data...")
    wid_df = pd.read_csv('modified/WID_percentiles_all_concepts.csv')
    print(f"Loaded WID data: {len(wid_df):,} rows, {wid_df['country'].nunique()} countries")

    # Prepare datasets for each source
    print("\n" + "="*70)
    print("DATASET 1: PIP (per capita)")
    print("="*70)
    pip_results = calculate_mld_decomposition(pip_df, 'average', 'pop')

    print(f"\nGlobal mean income: ${pip_results['global_mean']:.2f}")
    print(f"Total population: {pip_results['total_pop']:,.0f}")
    print(f"Number of countries: {pip_results['num_countries']}")
    print(f"\nTotal MLD: {pip_results['total_mld']:.4f}")
    print(f"Between-country: {pip_results['between_share']*100:.1f}%")
    print(f"Within-country: {pip_results['within_share']*100:.1f}%")

    # WID pre-tax per adult
    print("\n" + "="*70)
    print("DATASET 2: WID Pre-tax (per adult)")
    print("="*70)
    wid_pretax_adult_results = calculate_mld_decomposition(
        wid_df, 'avg_pretax_per_adult', 'bin_adult_pop'
    )

    print(f"\nGlobal mean income: ${wid_pretax_adult_results['global_mean']:.2f}")
    print(f"Total population: {wid_pretax_adult_results['total_pop']:,.0f}")
    print(f"Number of countries: {wid_pretax_adult_results['num_countries']}")
    print(f"\nTotal MLD: {wid_pretax_adult_results['total_mld']:.4f}")
    print(f"Between-country: {wid_pretax_adult_results['between_share']*100:.1f}%")
    print(f"Within-country: {wid_pretax_adult_results['within_share']*100:.1f}%")

    # WID pre-tax per capita
    print("\n" + "="*70)
    print("DATASET 3: WID Pre-tax (per capita)")
    print("="*70)
    wid_pretax_capita_results = calculate_mld_decomposition(
        wid_df, 'avg_pretax_per_capita', 'bin_total_pop'
    )

    print(f"\nGlobal mean income: ${wid_pretax_capita_results['global_mean']:.2f}")
    print(f"Total population: {wid_pretax_capita_results['total_pop']:,.0f}")
    print(f"Number of countries: {wid_pretax_capita_results['num_countries']}")
    print(f"\nTotal MLD: {wid_pretax_capita_results['total_mld']:.4f}")
    print(f"Between-country: {wid_pretax_capita_results['between_share']*100:.1f}%")
    print(f"Within-country: {wid_pretax_capita_results['within_share']*100:.1f}%")

    # WID post-tax per capita
    print("\n" + "="*70)
    print("DATASET 4: WID Post-tax (per capita)")
    print("="*70)
    wid_posttax_capita_results = calculate_mld_decomposition(
        wid_df, 'avg_posttax_per_capita', 'bin_total_pop'
    )

    print(f"\nGlobal mean income: ${wid_posttax_capita_results['global_mean']:.2f}")
    print(f"Total population: {wid_posttax_capita_results['total_pop']:,.0f}")
    print(f"Number of countries: {wid_posttax_capita_results['num_countries']}")
    print(f"\nTotal MLD: {wid_posttax_capita_results['total_mld']:.4f}")
    print(f"Between-country: {wid_posttax_capita_results['between_share']*100:.1f}%")
    print(f"Within-country: {wid_posttax_capita_results['within_share']*100:.1f}%")

    # Create visualization
    print("\n" + "="*70)
    print("Creating visualization...")
    print("="*70)

    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = [
        ('PIP\n(per capita)', pip_results),
        ('WID Pre-tax\n(per adult)', wid_pretax_adult_results),
        ('WID Pre-tax\n(per capita)', wid_pretax_capita_results),
        ('WID Post-tax\n(per capita)', wid_posttax_capita_results),
    ]

    x_pos = np.arange(len(datasets))
    width = 0.6

    # Colors
    between_color = '#2E86AB'
    within_color = '#A23B72'

    # Plot stacked bars
    between_values = [d[1]['between_share'] * 100 for d in datasets]
    within_values = [d[1]['within_share'] * 100 for d in datasets]

    ax.bar(x_pos, between_values, width, label='Between-country', color=between_color)
    ax.bar(x_pos, within_values, width, bottom=between_values, label='Within-country', color=within_color)

    # Add percentage labels
    for i, (label, results) in enumerate(datasets):
        between_pct = results['between_share'] * 100
        within_pct = results['within_share'] * 100

        # Between label
        ax.text(i, between_pct/2, f"{between_pct:.1f}%",
                ha='center', va='center', fontweight='bold', fontsize=10, color='white')

        # Within label
        ax.text(i, between_pct + within_pct/2, f"{within_pct:.1f}%",
                ha='center', va='center', fontweight='bold', fontsize=10, color='white')

    # Formatting
    ax.set_ylabel('Share of Total Inequality (%)', fontsize=12)
    ax.set_title('Global Income Inequality Decomposition\nMean Log Deviation (MLD) - Between vs Within Country',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d[0] for d in datasets], fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add note
    note_text = f"Note: Analysis based on {pip_results['num_countries']} countries (PIP) and {wid_pretax_capita_results['num_countries']} countries (WID) for year {year}"
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save
    output_file = 'outputs/income_concepts_decomposition.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Dataset':<30} {'Between':<12} {'Within':<12} {'Total MLD':<12}")
    print("-" * 70)
    for label, results in datasets:
        label_clean = label.replace('\n', ' ')
        print(f"{label_clean:<30} {results['between_share']*100:>6.1f}%      {results['within_share']*100:>6.1f}%      {results['total_mld']:>8.4f}")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
