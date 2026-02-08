"""
Compare within-country MLD (Mean Log Deviation) between PIP and WID data.

This script calculates the within-country inequality (MLD) for each individual
country using both PIP and WID (per capita) data, then creates an interactive
scatter plot to compare them.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go


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
        return np.nan

    total_pop = valid_df['pop'].sum()

    # Calculate MLD: Σ (n_i/N) * ln(μ/μ_i)
    mld = ((valid_df['pop'] / total_pop) * np.log(country_mean / valid_df['average'])).sum()

    return mld


def calculate_all_country_mlds(df):
    """
    Calculate MLD for each country in the dataset.

    Args:
        df: DataFrame with 'country', 'pop', 'average' columns

    Returns:
        DataFrame with country, mld, population, mean_income columns
    """
    results = []

    for country in df['country'].unique():
        country_df = df[df['country'] == country]
        country_pop = country_df['pop'].sum()
        country_mean = (country_df['pop'] * country_df['average']).sum() / country_pop

        country_mld = calculate_mld_within_country(country_df, country_mean)

        results.append({
            'country': country,
            'mld': country_mld,
            'population': country_pop,
            'mean_income': country_mean
        })

    return pd.DataFrame(results)


def create_interactive_mld_comparison(pip_mlds, wid_mlds, output_file='outputs/pip_vs_wid_country_mlds.html'):
    """
    Create interactive scatter plot comparing country-level MLDs between PIP and WID.

    Args:
        pip_mlds: DataFrame with country MLDs from PIP
        wid_mlds: DataFrame with country MLDs from WID
        output_file: path to save the HTML file
    """
    # Merge PIP and WID MLDs
    comparison = pip_mlds.merge(
        wid_mlds,
        on='country',
        suffixes=('_pip', '_wid'),
        how='inner'
    )

    # Remove any rows with missing MLD values
    comparison = comparison.dropna(subset=['mld_pip', 'mld_wid'])

    print(f"\nComparing {len(comparison)} countries with MLD data from both sources")

    # Identify top 10 most populous countries
    top10_countries = comparison.nlargest(10, 'population_wid')
    print(f"\nTop 10 most populous countries:")
    for idx, row in top10_countries.iterrows():
        print(f"  {row['country']}: Pop={row['population_wid']:,.0f}, PIP MLD={row['mld_pip']:.3f}, WID MLD={row['mld_wid']:.3f}")

    # Separate data
    top10_mask = comparison['country'].isin(top10_countries['country'])
    other_countries = comparison[~top10_mask]
    top10_data = comparison[top10_mask]

    # Create figure
    fig = go.Figure()

    # Set axis ranges
    # Use PIP range for x-axis, WID range for y-axis
    pip_min = comparison['mld_pip'].min() * 0.95
    pip_max = comparison['mld_pip'].max() * 1.05
    wid_min = comparison['mld_wid'].min() * 0.95
    wid_max = comparison['mld_wid'].max() * 1.05

    # Add reference lines (1x, 2x, 3x, 5x, 7x)
    multipliers = [1, 2, 3, 5, 7]
    x_range = np.array([pip_min, pip_max])

    reference_line_annotations = []

    for mult in multipliers:
        color = 'red' if mult == 1 else 'lightgray'
        dash = 'dash' if mult == 1 else 'dot'
        width = 2 if mult == 1 else 1.5

        label_text = 'WID = PIP' if mult == 1 else f'{mult}x'

        fig.add_trace(go.Scatter(
            x=x_range,
            y=x_range * mult,
            mode='lines',
            line=dict(color=color, dash=dash, width=width),
            name=label_text,
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add text annotation on the line (position at ~70% along x-axis)
        label_x = pip_min + 0.7 * (pip_max - pip_min)
        label_y = label_x * mult

        reference_line_annotations.append(
            dict(
                x=label_x,
                y=label_y,
                text=label_text,
                showarrow=False,
                font=dict(size=10, color=color if mult == 1 else 'gray'),
                bgcolor='rgba(255,255,255,0.7)',
                borderpad=2
            )
        )

    # Add scatter for other countries
    hover_text_other = []
    for idx, row in other_countries.iterrows():
        text = (f"<b>{row['country']}</b><br>"
                f"PIP MLD: {row['mld_pip']:.3f}<br>"
                f"WID MLD: {row['mld_wid']:.3f}<br>"
                f"Difference: {row['mld_wid'] - row['mld_pip']:.3f}<br>"
                f"Mean income (PIP): ${row['mean_income_pip']:.2f}/day<br>"
                f"Mean income (WID): ${row['mean_income_wid']:.2f}/day")
        hover_text_other.append(text)

    fig.add_trace(go.Scatter(
        x=other_countries['mld_pip'],
        y=other_countries['mld_wid'],
        mode='markers',
        marker=dict(size=8, color='#2E86AB', opacity=0.6, line=dict(color='white', width=0.5)),
        name='Countries',
        text=hover_text_other,
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add scatter for top 10 countries
    hover_text_top10 = []
    for idx, row in top10_data.iterrows():
        text = (f"<b>{row['country']}</b> (Pop: {row['population_wid']/1e9:.2f}B)<br>"
                f"PIP MLD: {row['mld_pip']:.3f}<br>"
                f"WID MLD: {row['mld_wid']:.3f}<br>"
                f"Difference: {row['mld_wid'] - row['mld_pip']:.3f}<br>"
                f"Mean income (PIP): ${row['mean_income_pip']:.2f}/day<br>"
                f"Mean income (WID): ${row['mean_income_wid']:.2f}/day")
        hover_text_top10.append(text)

    fig.add_trace(go.Scatter(
        x=top10_data['mld_pip'],
        y=top10_data['mld_wid'],
        mode='markers+text',
        marker=dict(size=15, color='#E63946', opacity=0.9, line=dict(color='black', width=1.5)),
        name='Top 10 most populous',
        text=top10_data['country'],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        customdata=hover_text_top10,
        hovertemplate='%{customdata}<extra></extra>'
    ))

    # Calculate correlation
    correlation = comparison['mld_pip'].corr(comparison['mld_wid'])

    # Calculate statistics
    mean_diff = (comparison['mld_wid'] - comparison['mld_pip']).mean()
    median_diff = (comparison['mld_wid'] - comparison['mld_pip']).median()

    # Update layout
    annotation_text = (f"Correlation: {correlation:.3f}<br>"
                      f"Mean difference (WID - PIP): {mean_diff:.3f}<br>"
                      f"Median difference: {median_diff:.3f}<br>"
                      f"n = {len(comparison)} countries")

    fig.update_layout(
        title=dict(
            text='Within-Country Inequality (MLD) by Country<br>PIP vs WID (2023, per capita)',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='PIP Within-Country MLD',
            gridcolor='lightgray',
            gridwidth=0.5,
            range=[pip_min, pip_max]
        ),
        yaxis=dict(
            title='WID Within-Country MLD',
            gridcolor='lightgray',
            gridwidth=0.5,
            range=[wid_min, wid_max]
        ),
        plot_bgcolor='white',
        hovermode='closest',
        width=1000,
        height=1000,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        annotations=[
            dict(
                text=annotation_text,
                xref='paper',
                yref='paper',
                x=0.98,
                y=0.02,
                xanchor='right',
                yanchor='bottom',
                showarrow=False,
                bgcolor='rgba(255,240,200,0.8)',
                bordercolor='black',
                borderwidth=1,
                borderpad=10,
                font=dict(size=11)
            )
        ] + reference_line_annotations
    )

    # Save to HTML
    fig.write_html(output_file)
    print(f"\nInteractive MLD comparison saved to: {output_file}")

    return comparison


def main():
    """Main execution function."""
    print("="*70)
    print("COUNTRY-LEVEL MLD COMPARISON: PIP vs WID")
    print("="*70)

    # Load harmonized data
    print("\nLoading harmonized data...")
    df = pd.read_csv('modified/pip_wid_harmonized.csv')
    print(f"Loaded {len(df):,} rows")

    # Filter to PIP and WID_per_capita
    pip_df = df[df['source'] == 'PIP'].copy()
    wid_df = df[df['source'] == 'WID_per_capita'].copy()

    print(f"\nPIP rows: {len(pip_df):,}")
    print(f"WID (per capita) rows: {len(wid_df):,}")

    # Calculate MLDs for each country
    print("\nCalculating within-country MLDs for PIP data...")
    pip_mlds = calculate_all_country_mlds(pip_df)
    print(f"Calculated MLDs for {len(pip_mlds)} countries")
    print(f"PIP MLD range: {pip_mlds['mld'].min():.3f} to {pip_mlds['mld'].max():.3f}")
    print(f"PIP MLD mean: {pip_mlds['mld'].mean():.3f}")

    print("\nCalculating within-country MLDs for WID data...")
    wid_mlds = calculate_all_country_mlds(wid_df)
    print(f"Calculated MLDs for {len(wid_mlds)} countries")
    print(f"WID MLD range: {wid_mlds['mld'].min():.3f} to {wid_mlds['mld'].max():.3f}")
    print(f"WID MLD mean: {wid_mlds['mld'].mean():.3f}")

    # Create interactive comparison plot
    print("\n" + "="*70)
    print("Creating interactive comparison plot...")
    comparison = create_interactive_mld_comparison(pip_mlds, wid_mlds)

    # Print some interesting statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print(f"\nCountries where WID MLD > PIP MLD: {(comparison['mld_wid'] > comparison['mld_pip']).sum()}")
    print(f"Countries where PIP MLD > WID MLD: {(comparison['mld_pip'] > comparison['mld_wid']).sum()}")

    print("\nCountries with largest WID MLD (top 5):")
    top_wid = comparison.nlargest(5, 'mld_wid')[['country', 'mld_pip', 'mld_wid']]
    for idx, row in top_wid.iterrows():
        print(f"  {row['country']}: PIP={row['mld_pip']:.3f}, WID={row['mld_wid']:.3f}")

    print("\nCountries with largest PIP MLD (top 5):")
    top_pip = comparison.nlargest(5, 'mld_pip')[['country', 'mld_pip', 'mld_wid']]
    for idx, row in top_pip.iterrows():
        print(f"  {row['country']}: PIP={row['mld_pip']:.3f}, WID={row['mld_wid']:.3f}")

    print("\nCountries with largest difference (WID - PIP, top 5):")
    comparison['diff'] = comparison['mld_wid'] - comparison['mld_pip']
    top_diff = comparison.nlargest(5, 'diff')[['country', 'mld_pip', 'mld_wid', 'diff']]
    for idx, row in top_diff.iterrows():
        print(f"  {row['country']}: PIP={row['mld_pip']:.3f}, WID={row['mld_wid']:.3f}, Diff={row['diff']:.3f}")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
