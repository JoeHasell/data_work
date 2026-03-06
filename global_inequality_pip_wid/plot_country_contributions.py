"""
Generate chart showing individual country contributions to within-country inequality.

This script breaks down the within-country MLD component by showing which countries
contribute the most to global within-country inequality. Countries contributing less
than a threshold are grouped into "Other countries".
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import MODIFIED_DIR, OUTPUTS_DIR
from mld_decomposition import MIN_INCOME

# Threshold for showing individual countries (as fraction of total within-country MLD)
CONTRIBUTION_THRESHOLD = 0.02  # Show countries contributing >2%


def calculate_country_contributions(df, income_col='average', pop_col='pop', country_col='country'):
    """
    Calculate each country's contribution to total within-country inequality.

    Returns DataFrame with columns: country, contribution, contribution_pct
    """
    # Make a copy and replace zeros
    df = df.copy()
    df.loc[df[income_col] <= 0, income_col] = MIN_INCOME

    # Calculate global statistics
    total_pop = df[pop_col].sum()

    # Calculate contribution for each country
    contributions = []
    for country in df[country_col].unique():
        country_df = df[df[country_col] == country].copy()

        # Country statistics
        country_mean = (country_df[income_col] * country_df[pop_col]).sum() / country_df[pop_col].sum()
        country_pop = country_df[pop_col].sum()
        pop_weight = country_pop / total_pop

        # Within-country MLD for this country
        country_mld = (
            (country_df[pop_col] / country_pop) *
            np.log(country_mean / country_df[income_col])
        ).sum()

        # Contribution to global within-country inequality
        contribution = pop_weight * country_mld

        contributions.append({
            'country': country,
            'contribution': contribution,
            'country_mld': country_mld,
            'pop_weight': pop_weight
        })

    contrib_df = pd.DataFrame(contributions)

    # Calculate percentage of total within-country inequality
    total_within = contrib_df['contribution'].sum()
    contrib_df['contribution_pct'] = contrib_df['contribution'] / total_within

    # Sort by contribution
    contrib_df = contrib_df.sort_values('contribution', ascending=False)

    return contrib_df, total_within


def calculate_between_mld(df, income_col='average', pop_col='pop', country_col='country'):
    """Calculate between-country MLD component."""
    # Make a copy and replace zeros
    df = df.copy()
    df.loc[df[income_col] <= 0, income_col] = MIN_INCOME

    # Calculate global mean
    total_income = (df[income_col] * df[pop_col]).sum()
    total_pop = df[pop_col].sum()
    global_mean = total_income / total_pop

    # Calculate country means
    country_stats = df.groupby(country_col).apply(
        lambda x: pd.Series({
            'country_mean': (x[income_col] * x[pop_col]).sum() / x[pop_col].sum(),
            'country_pop': x[pop_col].sum()
        })
    ).reset_index()

    # Between-country MLD
    between_mld = (
        (country_stats['country_pop'] / total_pop) *
        np.log(global_mean / country_stats['country_mean'])
    ).sum()

    return between_mld


def create_contribution_chart():
    """
    Create stacked bar chart showing between-country MLD and breakdown of within-country
    contributions by individual countries.
    """
    # Load harmonized data
    print("Loading harmonized data...")
    df = pd.read_csv(f"{MODIFIED_DIR}/pip_wid_harmonized.csv")

    # Filter to 2023 and relevant sources
    df = df[df['year'] == 2023]

    # Calculate contributions for PIP and WID per capita
    print("\nCalculating country contributions for PIP...")
    pip_df = df[df['source'] == 'PIP'].copy()
    pip_contrib, pip_total_within = calculate_country_contributions(pip_df)
    pip_between = calculate_between_mld(pip_df)

    print("\nCalculating country contributions for WID per capita...")
    wid_df = df[df['source'] == 'WID_per_capita'].copy()
    wid_contrib, wid_total_within = calculate_country_contributions(wid_df)
    wid_between = calculate_between_mld(wid_df)

    # Identify significant countries (above threshold in EITHER dataset)
    significant_countries = set(
        pip_contrib[pip_contrib['contribution_pct'] > CONTRIBUTION_THRESHOLD]['country'].tolist() +
        wid_contrib[wid_contrib['contribution_pct'] > CONTRIBUTION_THRESHOLD]['country'].tolist()
    )

    print(f"\nCountries contributing >{CONTRIBUTION_THRESHOLD*100}% to within-country inequality:")
    print(f"  {len(significant_countries)} countries: {', '.join(sorted(significant_countries))}")

    # Create grouped data for plotting
    def group_contributions(contrib_df, significant_countries):
        significant = contrib_df[contrib_df['country'].isin(significant_countries)].copy()
        other = contrib_df[~contrib_df['country'].isin(significant_countries)]

        # Sort significant countries by contribution
        significant = significant.sort_values('contribution', ascending=False)

        # Add "Other countries" row
        if len(other) > 0:
            other_row = pd.DataFrame([{
                'country': 'Other countries',
                'contribution': other['contribution'].sum(),
                'contribution_pct': other['contribution_pct'].sum()
            }])
            result = pd.concat([significant, other_row], ignore_index=True)
        else:
            result = significant

        return result

    pip_grouped = group_contributions(pip_contrib, significant_countries)
    wid_grouped = group_contributions(wid_contrib, significant_countries)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors
    between_color = '#3498db'  # Blue for between-country
    colors = plt.cm.Set3(np.linspace(0, 1, len(significant_countries) + 1))
    country_colors = {}
    for i, country in enumerate(sorted(significant_countries)):
        country_colors[country] = colors[i]
    country_colors['Other countries'] = '#cccccc'  # Gray for "Other"

    # Bar positions
    bar_width = 0.35
    x_positions = np.array([0, 1])

    # Plot PIP stacked bars
    # Start with between-country component
    ax.bar(x_positions[0], pip_between, bar_width,
           color=between_color, edgecolor='white', linewidth=1, label='Between-country')
    ax.text(x_positions[0], pip_between / 2, 'Between-\ncountry', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Then add within-country contributions
    bottom_pip = pip_between
    for idx, row in pip_grouped.iterrows():
        country = row['country']
        value = row['contribution']
        ax.bar(x_positions[0], value, bar_width, bottom=bottom_pip,
               color=country_colors[country], edgecolor='white', linewidth=1)

        # Label significant countries directly on the bar
        if country != 'Other countries' and row['contribution_pct'] > 0.03:
            label_y = bottom_pip + value / 2
            ax.text(x_positions[0], label_y, country, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='black')

        bottom_pip += value

    # Plot WID stacked bars
    # Start with between-country component
    ax.bar(x_positions[1], wid_between, bar_width,
           color=between_color, edgecolor='white', linewidth=1)
    ax.text(x_positions[1], wid_between / 2, 'Between-\ncountry', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Then add within-country contributions
    bottom_wid = wid_between
    for idx, row in wid_grouped.iterrows():
        country = row['country']
        value = row['contribution']
        ax.bar(x_positions[1], value, bar_width, bottom=bottom_wid,
               color=country_colors[country], edgecolor='white', linewidth=1)

        # Label significant countries directly on the bar
        if country != 'Other countries' and row['contribution_pct'] > 0.03:
            label_y = bottom_wid + value / 2
            ax.text(x_positions[1], label_y, country, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='black')

        bottom_wid += value

    # Formatting
    ax.set_ylabel('Mean Log Deviation (MLD)', fontsize=12, fontweight='bold')
    ax.set_title('Global Inequality Decomposition: Between-Country vs Within-Country',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['PIP\n(per capita)', 'WID\n(per capita)'], fontsize=11)
    ax.set_ylim(0, max(bottom_pip, bottom_wid) * 1.05)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add minimal legend
    between_patch = mpatches.Patch(color=between_color, label='Between-country')
    other_patch = mpatches.Patch(color='#cccccc', label='Other countries')
    ax.legend(handles=[between_patch, other_patch], loc='upper right', fontsize=10)

    # Add text annotations showing totals and percentages
    pip_total = bottom_pip
    wid_total = bottom_wid
    pip_between_pct = (pip_between / pip_total) * 100
    pip_within_pct = (pip_total_within / pip_total) * 100
    wid_between_pct = (wid_between / wid_total) * 100
    wid_within_pct = (wid_total_within / wid_total) * 100

    ax.text(x_positions[0], pip_total * 1.02,
           f'Total: {pip_total:.2f}\n({pip_between_pct:.1f}% between, {pip_within_pct:.1f}% within)',
           ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(x_positions[1], wid_total * 1.02,
           f'Total: {wid_total:.2f}\n({wid_between_pct:.1f}% between, {wid_within_pct:.1f}% within)',
           ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = f"{OUTPUTS_DIR}/country_contributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    pip_total = pip_between + pip_total_within
    wid_total = wid_between + wid_total_within

    print(f"\nPIP:")
    print(f"  Between-country MLD: {pip_between:.4f} ({pip_between/pip_total*100:.1f}%)")
    print(f"  Within-country MLD:  {pip_total_within:.4f} ({pip_total_within/pip_total*100:.1f}%)")
    print(f"  Total MLD:           {pip_total:.4f}")
    print("\n  Top 5 within-country contributors:")
    for idx, row in pip_grouped.head(5).iterrows():
        print(f"    {row['country']:20s}: {row['contribution']:.4f} ({row['contribution_pct']*100:.1f}%)")

    print(f"\nWID (per capita):")
    print(f"  Between-country MLD: {wid_between:.4f} ({wid_between/wid_total*100:.1f}%)")
    print(f"  Within-country MLD:  {wid_total_within:.4f} ({wid_total_within/wid_total*100:.1f}%)")
    print(f"  Total MLD:           {wid_total:.4f}")
    print("\n  Top 5 within-country contributors:")
    for idx, row in wid_grouped.head(5).iterrows():
        print(f"    {row['country']:20s}: {row['contribution']:.4f} ({row['contribution_pct']*100:.1f}%)")


if __name__ == "__main__":
    create_contribution_chart()
