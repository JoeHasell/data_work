"""
Counterfactual inequality analysis: Swap inequality distributions between countries.

This script allows us to take the inequality distribution (shape) from one country
and apply it to another country's mean income, then analyze the impact on global
inequality decomposition.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_country_mean(country_df):
    """Calculate mean income for a country."""
    return (country_df['pop'] * country_df['average']).sum() / country_df['pop'].sum()


def swap_inequality_distribution(df, source_col, target_country, source_country, source_type='WID_per_capita'):
    """
    Create counterfactual: Apply source country's inequality distribution to target country.

    Args:
        df: Full harmonized DataFrame
        source_col: Source column identifier (e.g., 'WID_per_capita')
        target_country: Country to modify (e.g., 'Mexico')
        source_country: Country to take distribution from (e.g., 'United States')
        source_type: Which data source to use

    Returns:
        Modified DataFrame with counterfactual for target country
    """
    df_modified = df.copy()

    # Get target and source country data
    target_data = df[(df['country'] == target_country) & (df['source'] == source_type)].copy()
    source_data = df[(df['country'] == source_country) & (df['source'] == source_type)].copy()

    if len(target_data) == 0:
        print(f"Warning: No data found for {target_country} with source {source_type}")
        return df_modified

    if len(source_data) == 0:
        print(f"Warning: No data found for {source_country} with source {source_type}")
        return df_modified

    # Calculate means
    target_mean = calculate_country_mean(target_data)
    source_mean = calculate_country_mean(source_data)

    print(f"\n{target_country} original mean: ${target_mean:.2f}/day")
    print(f"{source_country} mean: ${source_mean:.2f}/day")

    # Calculate rescaling factor
    rescale_factor = target_mean / source_mean
    print(f"Rescaling factor: {rescale_factor:.4f}")

    # Create counterfactual data for target country
    # Take source country's distribution shape, rescale to target's mean
    counterfactual = source_data.copy()
    counterfactual['country'] = target_country
    counterfactual['average'] = counterfactual['average'] * rescale_factor

    # Use target country's population (keep population structure the same)
    # Match percentiles between source and target
    for idx, row in counterfactual.iterrows():
        matching_target = target_data[
            (target_data['p_low'] == row['p_low']) &
            (target_data['p_high'] == row['p_high'])
        ]
        if len(matching_target) > 0:
            counterfactual.loc[idx, 'pop'] = matching_target.iloc[0]['pop']

    # Verify the mean is preserved
    counterfactual_mean = calculate_country_mean(counterfactual)
    print(f"{target_country} counterfactual mean: ${counterfactual_mean:.2f}/day")

    # Replace target country data with counterfactual
    df_modified = df_modified[
        ~((df_modified['country'] == target_country) & (df_modified['source'] == source_type))
    ]
    df_modified = pd.concat([df_modified, counterfactual], ignore_index=True)

    return df_modified


def calculate_mld_within_country(country_df, country_mean):
    """Calculate MLD within a single country."""
    valid_df = country_df[country_df['average'] > 0].copy()

    if len(valid_df) == 0 or country_mean <= 0:
        return 0.0

    total_pop = valid_df['pop'].sum()
    mld = ((valid_df['pop'] / total_pop) * np.log(country_mean / valid_df['average'])).sum()

    return mld


def decompose_mld(df):
    """Decompose MLD into between-country and within-country components."""
    # Calculate overall mean income
    global_mean = (df['pop'] * df['average']).sum() / df['pop'].sum()
    total_pop = df['pop'].sum()

    # Calculate country-level statistics
    country_stats = df.groupby('country').apply(
        lambda x: pd.Series({
            'population': x['pop'].sum(),
            'mean_income': (x['pop'] * x['average']).sum() / x['pop'].sum()
        })
    ).reset_index()

    # Between-country inequality
    valid_countries = country_stats[country_stats['mean_income'] > 0].copy()
    mld_between = (
        (valid_countries['population'] / total_pop) *
        np.log(global_mean / valid_countries['mean_income'])
    ).sum()

    # Within-country inequality
    mld_within = 0.0
    for country in valid_countries['country']:
        country_df = df[df['country'] == country]
        country_pop = country_df['pop'].sum()
        country_mean = (country_df['pop'] * country_df['average']).sum() / country_pop

        country_mld = calculate_mld_within_country(country_df, country_mean)
        mld_within += (country_pop / total_pop) * country_mld

    mld_total = mld_between + mld_within

    return {
        'total': mld_total,
        'between': mld_between,
        'within': mld_within
    }


def create_comparison_chart(results_dict, output_file='outputs/counterfactual_decomposition.png'):
    """
    Create stacked bar chart comparing MLD decomposition across scenarios.

    Args:
        results_dict: Dictionary with scenario names as keys, MLD results as values
        output_file: Path to save the chart
    """
    scenarios = list(results_dict.keys())
    between = [results_dict[s]['between'] for s in scenarios]
    within = [results_dict[s]['within'] for s in scenarios]
    totals = [results_dict[s]['total'] for s in scenarios]

    # Calculate percentages
    between_pcts = [(results_dict[s]['between'] / results_dict[s]['total']) * 100 for s in scenarios]
    within_pcts = [(results_dict[s]['within'] / results_dict[s]['total']) * 100 for s in scenarios]

    # Create figure (wider to accommodate more bars)
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create stacked bars
    x = np.arange(len(scenarios))
    width = 0.6

    # Plot bars
    bars_between = ax.bar(x, between, width, label='Between-country', color='#2E86AB')
    bars_within = ax.bar(x, within, width, bottom=between, label='Within-country', color='#A23B72')

    # Add annotations - Between-country
    for i, (bar, val, pct) in enumerate(zip(bars_between, between, between_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                f'{val:.3f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Add annotations - Within-country
    for i, (bar, val, pct) in enumerate(zip(bars_within, within, within_pcts)):
        height = bar.get_height()
        bottom = between[i]
        ax.text(bar.get_x() + bar.get_width() / 2., bottom + height / 2,
                f'{val:.3f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Add total MLD values on top
    for i, (x_pos, total) in enumerate(zip(x, totals)):
        ax.text(x_pos, total + 0.02,
                f'Total: {total:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Mean Log Deviation (MLD)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Counterfactual Inequality Distributions\nGlobal MLD Decomposition (WID per capita)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis limit with some padding
    max_total = max(totals)
    ax.set_ylim(0, max_total * 1.15)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCounterfactual chart saved to: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("COUNTERFACTUAL ANALYSIS: SWAPPING INEQUALITY DISTRIBUTIONS")
    print("="*70)

    # Load harmonized data
    print("\nLoading harmonized data...")
    df = pd.read_csv('modified/pip_wid_harmonized.csv')
    print(f"Loaded {len(df):,} rows")

    # Filter to WID per capita data
    wid_per_capita = df[df['source'] == 'WID_per_capita'].copy()
    print(f"WID per capita rows: {len(wid_per_capita):,}")

    # Analyze original WID per capita
    print("\n" + "="*70)
    print("ORIGINAL WID PER CAPITA")
    print("="*70)
    original_results = decompose_mld(wid_per_capita)
    print(f"\nTotal MLD:           {original_results['total']:.4f}")
    print(f"Between-country MLD: {original_results['between']:.4f} ({original_results['between']/original_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {original_results['within']:.4f} ({original_results['within']/original_results['total']*100:.1f}%)")

    # Store all results
    results_dict = {
        'WID per capita\n(Original)': original_results
    }

    # Counterfactual 1: Mexico with US inequality
    print("\n" + "="*70)
    print("COUNTERFACTUAL 1: MEXICO WITH US INEQUALITY DISTRIBUTION")
    print("="*70)

    df_cf1 = swap_inequality_distribution(
        df,
        source_col='WID_per_capita',
        target_country='Mexico',
        source_country='United States',
        source_type='WID_per_capita'
    )
    wid_cf1 = df_cf1[df_cf1['source'] == 'WID_per_capita'].copy()
    cf1_results = decompose_mld(wid_cf1)
    print(f"\nTotal MLD:           {cf1_results['total']:.4f}")
    print(f"Between-country MLD: {cf1_results['between']:.4f} ({cf1_results['between']/cf1_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {cf1_results['within']:.4f} ({cf1_results['within']/cf1_results['total']*100:.1f}%)")
    print(f"Change in between-country share: {(cf1_results['between']/cf1_results['total'] - original_results['between']/original_results['total'])*100:.2f} pp")

    results_dict['Mexico->US'] = cf1_results

    # Counterfactual 2: Brazil with US inequality
    print("\n" + "="*70)
    print("COUNTERFACTUAL 2: BRAZIL WITH US INEQUALITY DISTRIBUTION")
    print("="*70)

    df_cf2 = swap_inequality_distribution(
        df,
        source_col='WID_per_capita',
        target_country='Brazil',
        source_country='United States',
        source_type='WID_per_capita'
    )
    wid_cf2 = df_cf2[df_cf2['source'] == 'WID_per_capita'].copy()
    cf2_results = decompose_mld(wid_cf2)
    print(f"\nTotal MLD:           {cf2_results['total']:.4f}")
    print(f"Between-country MLD: {cf2_results['between']:.4f} ({cf2_results['between']/cf2_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {cf2_results['within']:.4f} ({cf2_results['within']/cf2_results['total']*100:.1f}%)")
    print(f"Change in between-country share: {(cf2_results['between']/cf2_results['total'] - original_results['between']/original_results['total'])*100:.2f} pp")

    results_dict['Brazil->US'] = cf2_results

    # Counterfactual 3: India with Pakistan inequality
    print("\n" + "="*70)
    print("COUNTERFACTUAL 3: INDIA WITH PAKISTAN INEQUALITY DISTRIBUTION")
    print("="*70)

    df_cf3 = swap_inequality_distribution(
        df,
        source_col='WID_per_capita',
        target_country='India',
        source_country='Pakistan',
        source_type='WID_per_capita'
    )
    wid_cf3 = df_cf3[df_cf3['source'] == 'WID_per_capita'].copy()
    cf3_results = decompose_mld(wid_cf3)
    print(f"\nTotal MLD:           {cf3_results['total']:.4f}")
    print(f"Between-country MLD: {cf3_results['between']:.4f} ({cf3_results['between']/cf3_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {cf3_results['within']:.4f} ({cf3_results['within']/cf3_results['total']*100:.1f}%)")
    print(f"Change in between-country share: {(cf3_results['between']/cf3_results['total'] - original_results['between']/original_results['total'])*100:.2f} pp")

    results_dict['India->Pakistan'] = cf3_results

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    for scenario, results in results_dict.items():
        change_bw_share = 0 if scenario == 'WID per capita\n(Original)' else \
            (results['between']/results['total'] - original_results['between']/original_results['total'])*100
        print(f"\n{scenario}:")
        print(f"  Total: {results['total']:.4f}, Between: {results['between']/results['total']*100:.1f}%, Change: {change_bw_share:+.2f} pp")

    # Create comparison chart
    create_comparison_chart(results_dict)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
