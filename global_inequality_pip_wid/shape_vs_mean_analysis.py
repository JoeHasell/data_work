"""
Analyze whether PIP vs WID differences are driven by mean incomes or distribution shapes.

This script creates counterfactuals that swap distribution shapes between PIP and WID
while preserving each source's mean incomes, helping isolate the impact of:
1. Different mean incomes vs
2. Different inequality patterns (distribution shapes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_country_mean(country_df):
    """Calculate mean income for a country."""
    return (country_df['pop'] * country_df['average']).sum() / country_df['pop'].sum()


def rescale_distribution_to_mean(df, target_source, new_mean_source):
    """
    Take distribution shapes from target_source and rescale to match means from new_mean_source.

    Args:
        df: Full harmonized DataFrame
        target_source: Source to take distribution shapes from (e.g., 'WID_per_capita')
        new_mean_source: Source to take mean incomes from (e.g., 'PIP')

    Returns:
        DataFrame with rescaled distributions
    """
    result_rows = []

    # Get unique countries present in both sources
    target_countries = set(df[df['source'] == target_source]['country'].unique())
    mean_countries = set(df[df['source'] == new_mean_source]['country'].unique())
    common_countries = target_countries & mean_countries

    print(f"\nRescaling {target_source} shapes to {new_mean_source} means")
    print(f"Countries in {target_source}: {len(target_countries)}")
    print(f"Countries in {new_mean_source}: {len(mean_countries)}")
    print(f"Common countries: {len(common_countries)}")

    # Process each country
    for country in common_countries:
        # Get distribution shape from target source
        target_data = df[(df['country'] == country) & (df['source'] == target_source)].copy()

        # Get mean from new mean source
        mean_data = df[(df['country'] == country) & (df['source'] == new_mean_source)].copy()

        if len(target_data) == 0 or len(mean_data) == 0:
            continue

        # Calculate means
        target_mean = calculate_country_mean(target_data)
        new_mean = calculate_country_mean(mean_data)

        if target_mean <= 0 or new_mean <= 0:
            continue

        # Calculate rescaling factor
        rescale_factor = new_mean / target_mean

        # Rescale the distribution
        rescaled_data = target_data.copy()
        rescaled_data['average'] = rescaled_data['average'] * rescale_factor

        # Keep the target source's population (important for WID vs PIP per capita difference)
        # This preserves the percentile structure from the target source

        result_rows.append(rescaled_data)

    if not result_rows:
        return pd.DataFrame()

    result_df = pd.concat(result_rows, ignore_index=True)

    # Verify rescaling worked
    sample_country = list(common_countries)[0]
    sample_rescaled = result_df[result_df['country'] == sample_country]
    sample_target = df[(df['country'] == sample_country) & (df['source'] == new_mean_source)]

    if len(sample_rescaled) > 0 and len(sample_target) > 0:
        rescaled_mean = calculate_country_mean(sample_rescaled)
        target_mean = calculate_country_mean(sample_target)
        print(f"Verification ({sample_country}): rescaled mean = ${rescaled_mean:.2f}, target mean = ${target_mean:.2f}")

    return result_df


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


def create_comparison_chart(results_dict, output_file='outputs/shape_vs_mean_decomposition.png'):
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

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create stacked bars
    x = np.arange(len(scenarios))
    width = 0.65

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
    ax.set_title('Shape vs Mean: Decomposing PIP-WID Differences\nGlobal MLD Decomposition (2023)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9, rotation=0, ha='center')
    ax.legend(fontsize=11, loc='upper right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis limit with some padding
    max_total = max(totals)
    ax.set_ylim(0, max_total * 1.15)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("SHAPE vs MEAN ANALYSIS: What Drives PIP vs WID Differences?")
    print("="*80)

    # Load harmonized data
    print("\nLoading harmonized data...")
    df = pd.read_csv('modified/pip_wid_harmonized.csv')
    print(f"Loaded {len(df):,} rows")

    # Filter to PIP and WID per capita for comparison
    pip_data = df[df['source'] == 'PIP'].copy()
    wid_data = df[df['source'] == 'WID_per_capita'].copy()

    print(f"\nPIP rows: {len(pip_data):,}")
    print(f"WID per capita rows: {len(wid_data):,}")

    # Analyze originals
    print("\n" + "="*80)
    print("ORIGINAL PIP (per capita)")
    print("="*80)
    pip_results = decompose_mld(pip_data)
    print(f"\nTotal MLD:           {pip_results['total']:.4f}")
    print(f"Between-country MLD: {pip_results['between']:.4f} ({pip_results['between']/pip_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {pip_results['within']:.4f} ({pip_results['within']/pip_results['total']*100:.1f}%)")

    print("\n" + "="*80)
    print("ORIGINAL WID (per capita)")
    print("="*80)
    wid_results = decompose_mld(wid_data)
    print(f"\nTotal MLD:           {wid_results['total']:.4f}")
    print(f"Between-country MLD: {wid_results['between']:.4f} ({wid_results['between']/wid_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {wid_results['within']:.4f} ({wid_results['within']/wid_results['total']*100:.1f}%)")

    # Counterfactual 1: WID shapes with PIP means
    print("\n" + "="*80)
    print("COUNTERFACTUAL 1: WID Distribution Shapes with PIP Mean Incomes")
    print("="*80)
    print("(Tests if distribution shape matters more than mean)")

    wid_shapes_pip_means = rescale_distribution_to_mean(df, 'WID_per_capita', 'PIP')
    cf1_results = decompose_mld(wid_shapes_pip_means)
    print(f"\nTotal MLD:           {cf1_results['total']:.4f}")
    print(f"Between-country MLD: {cf1_results['between']:.4f} ({cf1_results['between']/cf1_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {cf1_results['within']:.4f} ({cf1_results['within']/cf1_results['total']*100:.1f}%)")
    print(f"\nDifference from PIP between-country share: {(cf1_results['between']/cf1_results['total'] - pip_results['between']/pip_results['total'])*100:.2f} pp")
    print(f"Difference from WID between-country share: {(cf1_results['between']/cf1_results['total'] - wid_results['between']/wid_results['total'])*100:.2f} pp")

    # Counterfactual 2: PIP shapes with WID means
    print("\n" + "="*80)
    print("COUNTERFACTUAL 2: PIP Distribution Shapes with WID Mean Incomes")
    print("="*80)
    print("(Tests if mean income matters more than shape)")

    pip_shapes_wid_means = rescale_distribution_to_mean(df, 'PIP', 'WID_per_capita')
    cf2_results = decompose_mld(pip_shapes_wid_means)
    print(f"\nTotal MLD:           {cf2_results['total']:.4f}")
    print(f"Between-country MLD: {cf2_results['between']:.4f} ({cf2_results['between']/cf2_results['total']*100:.1f}%)")
    print(f"Within-country MLD:  {cf2_results['within']:.4f} ({cf2_results['within']/cf2_results['total']*100:.1f}%)")
    print(f"\nDifference from PIP between-country share: {(cf2_results['between']/cf2_results['total'] - pip_results['between']/pip_results['total'])*100:.2f} pp")
    print(f"Difference from WID between-country share: {(cf2_results['between']/cf2_results['total'] - wid_results['between']/wid_results['total'])*100:.2f} pp")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: What Drives the Difference?")
    print("="*80)

    original_gap = (pip_results['between']/pip_results['total'] - wid_results['between']/wid_results['total'])*100
    print(f"\nOriginal gap (PIP - WID): {original_gap:.2f} percentage points")
    print(f"  PIP: {pip_results['between']/pip_results['total']*100:.1f}% between-country")
    print(f"  WID: {wid_results['between']/wid_results['total']*100:.1f}% between-country")

    cf1_vs_pip = (cf1_results['between']/cf1_results['total'] - pip_results['between']/pip_results['total'])*100
    cf1_vs_wid = (cf1_results['between']/cf1_results['total'] - wid_results['between']/wid_results['total'])*100

    print(f"\nWID shapes with PIP means:")
    print(f"  Between-country share: {cf1_results['between']/cf1_results['total']*100:.1f}%")
    print(f"  Gap from PIP: {cf1_vs_pip:.2f} pp (closer = shape matters more)")
    print(f"  Gap from WID: {cf1_vs_wid:.2f} pp (closer = mean matters more)")

    cf2_vs_pip = (cf2_results['between']/cf2_results['total'] - pip_results['between']/pip_results['total'])*100
    cf2_vs_wid = (cf2_results['between']/cf2_results['total'] - wid_results['between']/wid_results['total'])*100

    print(f"\nPIP shapes with WID means:")
    print(f"  Between-country share: {cf2_results['between']/cf2_results['total']*100:.1f}%")
    print(f"  Gap from PIP: {cf2_vs_pip:.2f} pp (closer = mean matters more)")
    print(f"  Gap from WID: {cf2_vs_wid:.2f} pp (closer = shape matters more)")

    # Quantify the contribution
    print("\n" + "="*80)
    print("DECOMPOSITION OF THE GAP")
    print("="*80)

    # The gap can be decomposed into:
    # Gap = (PIP - WID) = (PIP - CF1) + (CF1 - WID)
    # where CF1 = WID shapes with PIP means
    # (PIP - CF1) = effect of different shapes given PIP means
    # (CF1 - WID) = effect of different means given WID shapes

    shape_effect = cf1_vs_pip  # Effect of shape difference given PIP means
    mean_effect = cf1_vs_wid    # Effect of mean difference given WID shapes

    print(f"\nUsing CF1 (WID shapes with PIP means) as bridge:")
    print(f"  Effect of different shapes (PIP shape - WID shape | PIP means): {shape_effect:.2f} pp")
    print(f"  Effect of different means (PIP means - WID means | WID shapes): {mean_effect:.2f} pp")
    print(f"  Sum: {shape_effect + mean_effect:.2f} pp (should equal total gap: {original_gap:.2f} pp)")

    # Create results dictionary
    results_dict = {
        'PIP\n(original)': pip_results,
        'WID\n(original)': wid_results,
        'WID shapes\nPIP means': cf1_results,
        'PIP shapes\nWID means': cf2_results
    }

    # Create comparison chart
    create_comparison_chart(results_dict)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
