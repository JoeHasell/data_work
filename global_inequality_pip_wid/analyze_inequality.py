"""
Analyze between-country vs within-country inequality using Mean Log Deviation (MLD).

This script calculates MLD for both PIP and WID data, decomposing it into
between-country and within-country components, and visualizes the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_mean_income(df):
    """
    Calculate overall mean income weighted by population.

    Args:
        df: DataFrame with 'pop' and 'average' columns

    Returns:
        float: Population-weighted mean income
    """
    total_income = (df['pop'] * df['average']).sum()
    total_pop = df['pop'].sum()
    return total_income / total_pop


def calculate_country_mean_income(df):
    """
    Calculate mean income for each country.

    Args:
        df: DataFrame with 'country', 'pop', and 'average' columns

    Returns:
        DataFrame: country, population, mean_income
    """
    country_stats = df.groupby('country').apply(
        lambda x: pd.Series({
            'population': x['pop'].sum(),
            'mean_income': (x['pop'] * x['average']).sum() / x['pop'].sum()
        })
    ).reset_index()

    return country_stats


def calculate_mld_within_country(country_df, country_mean):
    """
    Calculate MLD within a single country.

    MLD = Σ (n_i/N) * ln(μ/μ_i)

    Args:
        country_df: DataFrame for a single country with 'pop' and 'average' columns
        country_mean: Mean income for this country

    Returns:
        float: MLD within this country
    """
    # Avoid log(0) by filtering out zero or very small incomes
    valid_df = country_df[country_df['average'] > 0].copy()

    if len(valid_df) == 0 or country_mean <= 0:
        return 0.0

    total_pop = valid_df['pop'].sum()

    # Calculate MLD: Σ (n_i/N) * ln(μ/μ_i)
    mld = ((valid_df['pop'] / total_pop) * np.log(country_mean / valid_df['average'])).sum()

    return mld


def decompose_mld(df):
    """
    Decompose MLD into between-country and within-country components.

    Args:
        df: DataFrame with 'country', 'pop', 'average' columns

    Returns:
        dict: Contains 'total', 'between', 'within' MLD values
    """
    # Calculate overall mean income
    global_mean = calculate_mean_income(df)
    total_pop = df['pop'].sum()

    print(f"\nGlobal mean income: ${global_mean:.2f}")
    print(f"Total population: {total_pop:,.0f}")

    # Calculate country-level statistics
    country_stats = calculate_country_mean_income(df)
    print(f"Number of countries: {len(country_stats)}")

    # Calculate between-country inequality
    # MLD_between = Σ (N_c/N) * ln(μ/μ_c)
    valid_countries = country_stats[country_stats['mean_income'] > 0].copy()

    mld_between = (
        (valid_countries['population'] / total_pop) *
        np.log(global_mean / valid_countries['mean_income'])
    ).sum()

    # Calculate within-country inequality
    # MLD_within = Σ (N_c/N) * MLD_c
    mld_within = 0.0

    for country in valid_countries['country']:
        country_df = df[df['country'] == country]
        country_pop = country_df['pop'].sum()
        country_mean = (country_df['pop'] * country_df['average']).sum() / country_pop

        country_mld = calculate_mld_within_country(country_df, country_mean)
        mld_within += (country_pop / total_pop) * country_mld

    # Total MLD
    mld_total = mld_between + mld_within

    return {
        'total': mld_total,
        'between': mld_between,
        'within': mld_within
    }


def create_visualization(pip_results, wid_per_adult_results, wid_per_capita_results, output_file='outputs/inequality_decomposition.png'):
    """
    Create a stacked bar chart showing MLD decomposition for PIP, WID (per adult), and WID (per capita).

    Args:
        pip_results: dict with 'between', 'within', 'total' for PIP
        wid_per_adult_results: dict with 'between', 'within', 'total' for WID per adult
        wid_per_capita_results: dict with 'between', 'within', 'total' for WID per capita
        output_file: path to save the chart
    """
    # Prepare data
    sources = ['PIP\n(per capita)', 'WID\n(per adult)', 'WID\n(per capita)']
    results_list = [pip_results, wid_per_adult_results, wid_per_capita_results]

    between = [r['between'] for r in results_list]
    within = [r['within'] for r in results_list]
    totals = [r['total'] for r in results_list]

    # Calculate percentages
    between_pcts = [(r['between'] / r['total']) * 100 for r in results_list]
    within_pcts = [(r['within'] / r['total']) * 100 for r in results_list]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create stacked bars
    x = np.arange(len(sources))
    width = 0.6

    # Plot bars
    bars_between = ax.bar(x, between, width, label='Between-country', color='#2E86AB')
    bars_within = ax.bar(x, within, width, bottom=between, label='Within-country', color='#A23B72')

    # Add annotations - Between-country
    for i, (bar, val, pct) in enumerate(zip(bars_between, between, between_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                f'{val:.3f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Add annotations - Within-country
    for i, (bar, val, pct) in enumerate(zip(bars_within, within, within_pcts)):
        height = bar.get_height()
        bottom = between[i]
        ax.text(bar.get_x() + bar.get_width() / 2., bottom + height / 2,
                f'{val:.3f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Add total MLD values on top
    for i, (x_pos, total) in enumerate(zip(x, totals)):
        ax.text(x_pos, total + 0.02,
                f'Total: {total:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Mean Log Deviation (MLD)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Data Source', fontsize=12, fontweight='bold')
    ax.set_title('Decomposition of Global Income Inequality\nBetween vs Within Country Components',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis limit with some padding
    max_total = max(totals)
    ax.set_ylim(0, max_total * 1.15)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("GLOBAL INCOME INEQUALITY DECOMPOSITION ANALYSIS")
    print("Mean Log Deviation (MLD) - Between vs Within Country Components")
    print("="*70)

    # Load harmonized data
    print("\nLoading harmonized data...")
    df = pd.read_csv('modified/pip_wid_harmonized.csv')
    print(f"Loaded {len(df):,} rows")

    # Separate data by source
    pip_df = df[df['source'] == 'PIP'].copy()
    wid_per_adult_df = df[df['source'] == 'WID_per_adult'].copy()
    wid_per_capita_df = df[df['source'] == 'WID_per_capita'].copy()

    print(f"PIP rows: {len(pip_df):,}")
    print(f"WID (per adult) rows: {len(wid_per_adult_df):,}")
    print(f"WID (per capita) rows: {len(wid_per_capita_df):,}")

    # Analyze PIP data
    print("\n" + "="*70)
    print("PIP DATA ANALYSIS (per capita)")
    print("="*70)
    pip_results = decompose_mld(pip_df)

    # Analyze WID per adult data
    print("\n" + "="*70)
    print("WID DATA ANALYSIS (per adult)")
    print("="*70)
    wid_per_adult_results = decompose_mld(wid_per_adult_df)

    # Analyze WID per capita data
    print("\n" + "="*70)
    print("WID DATA ANALYSIS (per capita)")
    print("="*70)
    wid_per_capita_results = decompose_mld(wid_per_capita_df)

    # Print summary results
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)

    print("\nPIP (per capita):")
    print(f"  Total MLD:           {pip_results['total']:.4f}")
    print(f"  Between-country MLD: {pip_results['between']:.4f} ({pip_results['between']/pip_results['total']*100:.1f}%)")
    print(f"  Within-country MLD:  {pip_results['within']:.4f} ({pip_results['within']/pip_results['total']*100:.1f}%)")

    print("\nWID (per adult):")
    print(f"  Total MLD:           {wid_per_adult_results['total']:.4f}")
    print(f"  Between-country MLD: {wid_per_adult_results['between']:.4f} ({wid_per_adult_results['between']/wid_per_adult_results['total']*100:.1f}%)")
    print(f"  Within-country MLD:  {wid_per_adult_results['within']:.4f} ({wid_per_adult_results['within']/wid_per_adult_results['total']*100:.1f}%)")

    print("\nWID (per capita):")
    print(f"  Total MLD:           {wid_per_capita_results['total']:.4f}")
    print(f"  Between-country MLD: {wid_per_capita_results['between']:.4f} ({wid_per_capita_results['between']/wid_per_capita_results['total']*100:.1f}%)")
    print(f"  Within-country MLD:  {wid_per_capita_results['within']:.4f} ({wid_per_capita_results['within']/wid_per_capita_results['total']*100:.1f}%)")

    # Compare sources
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nPIP vs WID (per adult):")
    print(f"  Difference in total inequality: {pip_results['total'] - wid_per_adult_results['total']:.4f}")
    print(f"  Difference in between-country share: {(pip_results['between']/pip_results['total'] - wid_per_adult_results['between']/wid_per_adult_results['total'])*100:.1f} percentage points")

    print(f"\nPIP vs WID (per capita):")
    print(f"  Difference in total inequality: {pip_results['total'] - wid_per_capita_results['total']:.4f}")
    print(f"  Difference in between-country share: {(pip_results['between']/pip_results['total'] - wid_per_capita_results['between']/wid_per_capita_results['total'])*100:.1f} percentage points")

    print(f"\nWID (per adult) vs WID (per capita):")
    print(f"  Difference in total inequality: {wid_per_adult_results['total'] - wid_per_capita_results['total']:.4f}")
    print(f"  Difference in between-country share: {(wid_per_adult_results['between']/wid_per_adult_results['total'] - wid_per_capita_results['between']/wid_per_capita_results['total'])*100:.1f} percentage points")

    # Create visualization
    print("\n" + "="*70)
    print("Creating visualization...")
    create_visualization(pip_results, wid_per_adult_results, wid_per_capita_results)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
