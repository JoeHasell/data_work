"""
Sensitivity analysis for zero income replacement value.

Tests three replacement methods:
1. Baseline: Replace zeros with $0.01/day
2. Alternative 1: Replace zeros with $0.10/day
3. Alternative 2: Replace zeros with lowest non-zero percentile in each country
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_mld_with_replacement(df, replacement_method='baseline'):
    """
    Calculate MLD decomposition with specified zero replacement method.

    Args:
        df: DataFrame with columns: country, year, p_low, p_high, pop, average, source
        replacement_method: 'baseline' ($0.01), 'alt1' ($0.10), or 'country_min'

    Returns:
        dict with decomposition results
    """
    df = df.copy()

    # Apply replacement method
    if replacement_method == 'baseline':
        # Replace zeros with $0.01
        df.loc[df['average'] <= 0, 'average'] = 0.01

    elif replacement_method == 'alt1':
        # Replace zeros with $0.10
        df.loc[df['average'] <= 0, 'average'] = 0.10

    elif replacement_method == 'country_min':
        # For each country, replace zeros with the lowest non-zero value
        for country in df['country'].unique():
            country_data = df[df['country'] == country]

            # Find lowest non-zero value for this country
            non_zero_values = country_data[country_data['average'] > 0]['average']

            if len(non_zero_values) > 0:
                min_non_zero = non_zero_values.min()
                # Replace zeros with this minimum
                df.loc[(df['country'] == country) & (df['average'] <= 0), 'average'] = min_non_zero
            else:
                # If all values are zero (shouldn't happen), use $0.01
                df.loc[(df['country'] == country) & (df['average'] <= 0), 'average'] = 0.01

    # Calculate global mean
    total_pop = df['pop'].sum()
    global_mean = (df['average'] * df['pop']).sum() / total_pop

    # Calculate between-country MLD
    country_means = df.groupby('country').apply(
        lambda x: (x['average'] * x['pop']).sum() / x['pop'].sum()
    )
    country_pops = df.groupby('country')['pop'].sum()

    between_mld = sum(
        (country_pops[c] / total_pop) * np.log(global_mean / country_means[c])
        for c in country_means.index
    )

    # Calculate within-country MLD for each country
    within_mld_total = 0
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        country_pop = country_data['pop'].sum()
        country_mean = (country_data['average'] * country_data['pop']).sum() / country_pop

        # MLD for this country
        country_mld = sum(
            (row['pop'] / country_pop) * np.log(country_mean / row['average'])
            for _, row in country_data.iterrows()
        )

        # Add weighted contribution to global within-country MLD
        within_mld_total += (country_pop / total_pop) * country_mld

    total_mld = between_mld + within_mld_total

    return {
        'total_mld': total_mld,
        'between_country_mld': between_mld,
        'within_country_mld': within_mld_total,
        'between_country_share': between_mld / total_mld,
        'within_country_share': within_mld_total / total_mld,
        'global_mean': global_mean
    }


def main():
    """Run sensitivity analysis."""
    print("="*70)
    print("ZERO REPLACEMENT SENSITIVITY ANALYSIS")
    print("="*70)

    # Load harmonized data
    df = pd.read_csv('modified/pip_wid_harmonized.csv')

    # Filter to 2023
    df = df[df['year'] == 2023]

    # Get PIP per capita data
    pip_data = df[df['source'] == 'PIP'].copy()

    # Get WID per capita data
    wid_data = df[df['source'] == 'WID_per_capita'].copy()

    print(f"\nPIP data: {len(pip_data):,} rows")
    print(f"WID per capita data: {len(wid_data):,} rows")

    # Calculate decompositions for each method
    methods = ['baseline', 'alt1', 'country_min']
    method_labels = {
        'baseline': '$0.01 replacement',
        'alt1': '$0.10 replacement',
        'country_min': 'Country min replacement'
    }

    results = []

    # Only analyze WID per capita (PIP has no zeros)
    print(f"\nWID per capita:")
    print("-" * 50)

    for method in methods:
        decomp = calculate_mld_with_replacement(wid_data, method)

        results.append({
            'source': 'WID_per_capita',
            'method': method,
            'method_label': method_labels[method],
            'total_mld': decomp['total_mld'],
            'between_mld': decomp['between_country_mld'],
            'within_mld': decomp['within_country_mld'],
            'between_pct': decomp['between_country_share'] * 100,
            'within_pct': decomp['within_country_share'] * 100
        })

        print(f"\n  {method_labels[method]}:")
        print(f"    Total MLD: {decomp['total_mld']:.4f}")
        print(f"    Between: {decomp['between_country_share']*100:.1f}%")
        print(f"    Within: {decomp['within_country_share']*100:.1f}%")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv('outputs/zero_replacement_sensitivity.csv', index=False)
    print(f"\n\nResults saved to: outputs/zero_replacement_sensitivity.csv")

    # Create plot
    print("\nGenerating plot...")
    create_sensitivity_plot(results_df)

    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)


def create_sensitivity_plot(results_df):
    """Create stacked bar chart showing sensitivity to zero replacement."""

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Prepare data for plotting
    datasets = []
    between_vals = []
    within_vals = []
    between_pcts = []
    within_pcts = []

    # Only WID per capita with 3 methods
    for method in ['baseline', 'alt1', 'country_min']:
        row = results_df[results_df['method'] == method].iloc[0]

        # Create label
        method_label = row['method_label'].replace(' replacement', '')
        label = f"WID (per capita)\n{method_label}"

        datasets.append(label)
        between_vals.append(row['between_mld'])
        within_vals.append(row['within_mld'])
        between_pcts.append(row['between_pct'])
        within_pcts.append(row['within_pct'])

    # Create stacked bars
    x = np.arange(len(datasets))
    width = 0.6

    # Plot bars
    bars_between = ax.bar(x, between_vals, width, label='Between-country', color='#2E86AB')
    bars_within = ax.bar(x, within_vals, width, bottom=between_vals, label='Within-country', color='#A23B72')

    # Add annotations - Between-country
    for i, (bar, val, pct) in enumerate(zip(bars_between, between_vals, between_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                f'{val:.2f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Add annotations - Within-country
    for i, (bar, val, pct) in enumerate(zip(bars_within, within_vals, within_pcts)):
        height = bar.get_height()
        bottom = between_vals[i]
        ax.text(bar.get_x() + bar.get_width() / 2., bottom + height / 2,
                f'{val:.2f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Customize plot
    ax.set_ylabel('Mean Log Deviation (MLD)', fontsize=12, fontweight='bold')
    ax.set_title('WID: Sensitivity to Zero Income Replacement Value (2023)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('outputs/zero_replacement_sensitivity.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/zero_replacement_sensitivity.png")
    plt.close()


if __name__ == "__main__":
    main()
