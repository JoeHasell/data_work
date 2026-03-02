"""
Plot all MLD decomposition charts from pre-calculated CSV.

This script reads from outputs/mld_decompositions.csv and generates all
visualization charts for the narrative presentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_basic_decomposition(df, output_file='outputs/basic_decomposition.png'):
    """
    Create stacked bar chart comparing PIP vs WID (per adult).

    Args:
        df: DataFrame with decomposition results
        output_file: Path to save the chart
    """
    print("\n" + "="*70)
    print("PLOT 1: Basic Decomposition (PIP vs WID per adult)")
    print("="*70)

    # Filter to PIP per capita and WID pre-tax per adult
    pip_row = df[df['analysis_id'] == 'PIP_per_capita'].iloc[0]
    wid_row = df[df['analysis_id'] == 'WID_pretax_per_adult'].iloc[0]

    # Extract values
    datasets = ['PIP\n(per capita)', 'WID\n(per adult)']
    between_vals = [pip_row['between_country_mld'], wid_row['between_country_mld']]
    within_vals = [pip_row['within_country_mld'], wid_row['within_country_mld']]
    between_pcts = [pip_row['between_country_share'] * 100, wid_row['between_country_share'] * 100]
    within_pcts = [pip_row['within_country_share'] * 100, wid_row['within_country_share'] * 100]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create stacked bars
    x = np.arange(len(datasets))
    width = 0.5

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
    ax.set_title('Global Inequality Decomposition (2023)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_income_concepts(df, output_file='outputs/income_concepts_decomposition.png'):
    """
    Create stacked bar chart comparing all income concepts.

    Args:
        df: DataFrame with decomposition results
        output_file: Path to save the chart
    """
    print("\n" + "="*70)
    print("PLOT 2: Income Concepts Comparison")
    print("="*70)

    # Get all four analyses
    analyses = [
        ('PIP_per_capita', 'PIP\n(per capita)'),
        ('WID_pretax_per_adult', 'WID Pre-tax\n(per adult)'),
        ('WID_pretax_per_capita', 'WID Pre-tax\n(per capita)'),
        ('WID_posttax_per_capita', 'WID Post-tax\n(per capita)')
    ]

    datasets = []
    between_vals = []
    within_vals = []
    between_pcts = []
    within_pcts = []

    for analysis_id, label in analyses:
        row = df[df['analysis_id'] == analysis_id].iloc[0]
        datasets.append(label)
        between_vals.append(row['between_country_mld'])
        within_vals.append(row['within_country_mld'])
        between_pcts.append(row['between_country_share'] * 100)
        within_pcts.append(row['within_country_share'] * 100)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create stacked bars
    x = np.arange(len(datasets))
    width = 0.65

    # Plot bars
    bars_between = ax.bar(x, between_vals, width, label='Between-country', color='#2E86AB')
    bars_within = ax.bar(x, within_vals, width, bottom=between_vals, label='Within-country', color='#A23B72')

    # Add annotations - Between-country
    for i, (bar, val, pct) in enumerate(zip(bars_between, between_vals, between_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                f'{val:.2f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Add annotations - Within-country
    for i, (bar, val, pct) in enumerate(zip(bars_within, within_vals, within_pcts)):
        height = bar.get_height()
        bottom = between_vals[i]
        ax.text(bar.get_x() + bar.get_width() / 2., bottom + height / 2,
                f'{val:.2f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Customize plot
    ax.set_ylabel('Mean Log Deviation (MLD)', fontsize=12, fontweight='bold')
    ax.set_title('Global Inequality: Comparing Income Concepts (2023)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_inequality_comparison(df, output_file='outputs/inequality_decomposition.png'):
    """
    Create stacked bar chart comparing PIP vs all three WID variants.

    Args:
        df: DataFrame with decomposition results
        output_file: Path to save the chart
    """
    print("\n" + "="*70)
    print("PLOT 3: Full Inequality Comparison")
    print("="*70)

    # Get all four analyses (same as income concepts)
    analyses = [
        ('PIP_per_capita', 'PIP\n(per capita)'),
        ('WID_pretax_per_adult', 'WID Pre-tax\n(per adult)'),
        ('WID_pretax_per_capita', 'WID Pre-tax\n(per capita)'),
        ('WID_posttax_per_capita', 'WID Post-tax\n(per capita)')
    ]

    datasets = []
    between_vals = []
    within_vals = []
    between_pcts = []
    within_pcts = []

    for analysis_id, label in analyses:
        row = df[df['analysis_id'] == analysis_id].iloc[0]
        datasets.append(label)
        between_vals.append(row['between_country_mld'])
        within_vals.append(row['within_country_mld'])
        between_pcts.append(row['between_country_share'] * 100)
        within_pcts.append(row['within_country_share'] * 100)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create stacked bars
    x = np.arange(len(datasets))
    width = 0.65

    # Plot bars
    bars_between = ax.bar(x, between_vals, width, label='Between-country', color='#2E86AB')
    bars_within = ax.bar(x, within_vals, width, bottom=between_vals, label='Within-country', color='#A23B72')

    # Add annotations - Between-country
    for i, (bar, val, pct) in enumerate(zip(bars_between, between_vals, between_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                f'{val:.2f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Add annotations - Within-country
    for i, (bar, val, pct) in enumerate(zip(bars_within, within_vals, within_pcts)):
        height = bar.get_height()
        bottom = between_vals[i]
        ax.text(bar.get_x() + bar.get_width() / 2., bottom + height / 2,
                f'{val:.2f}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Customize plot
    ax.set_ylabel('Mean Log Deviation (MLD)', fontsize=12, fontweight='bold')
    ax.set_title('Global Inequality Decomposition: PIP vs WID (2023)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("PLOT ALL MLD DECOMPOSITIONS")
    print("="*70)
    print("\nReading decomposition results from CSV...")

    # Load decomposition results
    df = pd.read_csv('outputs/mld_decompositions.csv')
    print(f"Loaded {len(df)} decomposition results")

    # Generate all plots
    plot_basic_decomposition(df)
    plot_income_concepts(df)
    plot_inequality_comparison(df)

    print("\n" + "="*70)
    print("ALL PLOTS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - outputs/basic_decomposition.png")
    print("  - outputs/income_concepts_decomposition.png")
    print("  - outputs/inequality_decomposition.png")
    print("="*70)


if __name__ == "__main__":
    main()
