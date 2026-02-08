"""
Compare PIP and WID data through visualizations.

This script creates comparison plots between PIP and WID data sources,
including scatter plots of mean income by country.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_country_means(df):
    """
    Calculate mean income for each country from the percentile data.

    Args:
        df: DataFrame with 'country', 'source', 'pop', and 'average' columns

    Returns:
        DataFrame with country, source, and mean_income columns
    """
    country_means = df.groupby(['country', 'source']).apply(
        lambda x: pd.Series({
            'mean_income': (x['pop'] * x['average']).sum() / x['pop'].sum(),
            'population': x['pop'].sum()
        })
    ).reset_index()

    return country_means


def create_scatter_plot(pip_wid_comparison, output_file='outputs/pip_vs_wid_mean_income.png'):
    """
    Create scatter plot comparing PIP vs WID mean income by country.

    Args:
        pip_wid_comparison: DataFrame with 'country', 'pip_mean', 'wid_mean', 'wid_population'
        output_file: path to save the plot
    """
    # Remove any countries with missing data
    plot_df = pip_wid_comparison.dropna().copy()

    print(f"\nPlotting {len(plot_df)} countries with data from both sources")

    # Identify the 10 most populous countries (by WID population)
    top10_countries = plot_df.nlargest(10, 'wid_population')
    print(f"\nTop 10 most populous countries:")
    for idx, row in top10_countries.iterrows():
        print(f"  {row['country']}: {row['wid_population']:,.0f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Separate data: top 10 most populous vs others
    top10_mask = plot_df['country'].isin(top10_countries['country'])
    other_countries = plot_df[~top10_mask]
    top10_data = plot_df[top10_mask]

    # Plot other countries (regular scatter)
    ax.scatter(other_countries['pip_mean'], other_countries['wid_mean'],
               alpha=0.6, s=50, color='#2E86AB', edgecolors='white', linewidth=0.5, zorder=3)

    # Plot top 10 countries with highlighting
    ax.scatter(top10_data['pip_mean'], top10_data['wid_mean'],
               alpha=0.9, s=150, color='#E63946', edgecolors='black', linewidth=1.5, zorder=4)

    # Add labels for top 10 countries
    for idx, row in top10_data.iterrows():
        ax.annotate(row['country'],
                   xy=(row['pip_mean'], row['wid_mean']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', edgecolor='black', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

    # Add reference lines (WID = 1x, 2x, 3x, 5x, 7x, 10x, 20x PIP)
    min_val = plot_df['pip_mean'].min() * 0.8
    max_val = plot_df['pip_mean'].max() * 1.2

    # Specific multipliers to show
    multipliers = [1, 2, 3, 5, 7, 10, 20]

    # Plot reference lines
    x_range = np.array([min_val, max_val])

    for mult in multipliers:
        if mult == 1:
            color = 'red'
            linestyle = '--'
            linewidth = 2
            alpha = 0.7
        else:
            color = 'gray'
            linestyle = ':'
            linewidth = 1.5
            alpha = 0.5

        ax.plot(x_range, x_range * mult,
                color=color, linestyle=linestyle, linewidth=linewidth,
                alpha=alpha, zorder=1)

        # Add text label on the line itself
        # Position label at ~70% along the x-axis range (in log space)
        label_x = np.exp(np.log(min_val) + 0.7 * (np.log(max_val) - np.log(min_val)))
        label_y = label_x * mult

        # Adjust label for the 1x line
        label_text = 'WID = PIP' if mult == 1 else f'{mult}x'
        fontweight = 'bold' if mult == 1 else 'normal'
        fontsize = 10 if mult == 1 else 9

        ax.text(label_x, label_y, label_text,
                fontsize=fontsize, fontweight=fontweight,
                color=color, alpha=0.9,
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.7))

    # Fit linear regression lines in log-log space
    X_log = np.log(plot_df[['pip_mean']].values)
    y_log = np.log(plot_df['wid_mean'].values)

    # Unweighted regression (in log space)
    reg_unweighted = LinearRegression()
    reg_unweighted.fit(X_log, y_log)
    y_log_pred_unweighted = reg_unweighted.predict(X_log)
    y_pred_unweighted = np.exp(y_log_pred_unweighted)

    # Weighted regression (by WID population, in log space)
    reg_weighted = LinearRegression()
    weights = plot_df['wid_population'].values
    reg_weighted.fit(X_log, y_log, sample_weight=weights)
    y_log_pred_weighted = reg_weighted.predict(X_log)
    y_pred_weighted = np.exp(y_log_pred_weighted)

    # Sort for plotting
    X = plot_df[['pip_mean']].values
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx].flatten()
    y_unweighted_sorted = y_pred_unweighted[sort_idx]
    y_weighted_sorted = y_pred_weighted[sort_idx]

    # Plot regression lines
    ax.plot(X_sorted, y_unweighted_sorted,
            color='blue', linestyle='-', linewidth=2.5, alpha=0.7,
            label='Unweighted regression', zorder=2)
    ax.plot(X_sorted, y_weighted_sorted,
            color='purple', linestyle='-', linewidth=2.5, alpha=0.7,
            label='Population-weighted regression', zorder=2)

    # Add regression equation annotations
    # In log-log space: log(WID) = slope * log(PIP) + intercept
    # In linear space: WID = exp(intercept) * PIP^slope
    slope_unw = reg_unweighted.coef_[0]
    intercept_unw = reg_unweighted.intercept_
    slope_w = reg_weighted.coef_[0]
    intercept_w = reg_weighted.intercept_

    print(f"\nRegression results (log-log space):")
    print(f"  Unweighted: WID = {np.exp(intercept_unw):.2f} * PIP^{slope_unw:.2f}")
    print(f"  Weighted:   WID = {np.exp(intercept_w):.2f} * PIP^{slope_w:.2f}")

    # Custom formatter for dollar values
    def dollar_formatter(x, pos):
        if x >= 1:
            return f'${x:.0f}'
        else:
            return f'${x:.1f}'

    ax.xaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

    # Set labels and title
    ax.set_xlabel('PIP Mean Income ($ per day, 2021 PPP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('WID Mean Income ($ per day, 2021 PPP)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Mean Income by Country (per capita)\nPIP vs WID (2023)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_axisbelow(True)

    # Add legend for regression lines
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

    # Make axes equal in log space
    ax.set_aspect('equal', adjustable='box')

    # Add text annotation with correlation
    correlation = plot_df['pip_mean'].corr(plot_df['wid_mean'])
    # Calculate correlation in log space for better representation
    log_correlation = np.corrcoef(np.log(plot_df['pip_mean']), np.log(plot_df['wid_mean']))[0, 1]

    ax.text(0.98, 0.02, f'Correlation: {correlation:.3f}\nLog correlation: {log_correlation:.3f}\nn = {len(plot_df)} countries',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_file}")
    plt.close()

    return correlation


def print_summary_statistics(pip_wid_comparison):
    """Print summary statistics about the comparison."""
    plot_df = pip_wid_comparison.dropna().copy()

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print(f"\nCountries in both datasets: {len(plot_df)}")
    print(f"Countries only in PIP: {pip_wid_comparison['pip_mean'].notna().sum() - len(plot_df)}")
    print(f"Countries only in WID: {pip_wid_comparison['wid_mean'].notna().sum() - len(plot_df)}")

    print("\nPIP Mean Income:")
    print(f"  Min:    ${plot_df['pip_mean'].min():.2f}/day")
    print(f"  Max:    ${plot_df['pip_mean'].max():.2f}/day")
    print(f"  Median: ${plot_df['pip_mean'].median():.2f}/day")
    print(f"  Mean:   ${plot_df['pip_mean'].mean():.2f}/day")

    print("\nWID Mean Income:")
    print(f"  Min:    ${plot_df['wid_mean'].min():.2f}/day")
    print(f"  Max:    ${plot_df['wid_mean'].max():.2f}/day")
    print(f"  Median: ${plot_df['wid_mean'].median():.2f}/day")
    print(f"  Mean:   ${plot_df['wid_mean'].mean():.2f}/day")

    print("\nDifference (WID - PIP):")
    plot_df['difference'] = plot_df['wid_mean'] - plot_df['pip_mean']
    plot_df['ratio'] = plot_df['wid_mean'] / plot_df['pip_mean']

    print(f"  Mean difference:    ${plot_df['difference'].mean():.2f}/day")
    print(f"  Median difference:  ${plot_df['difference'].median():.2f}/day")
    print(f"  Mean ratio (WID/PIP): {plot_df['ratio'].mean():.2f}")
    print(f"  Median ratio (WID/PIP): {plot_df['ratio'].median():.2f}")

    # Countries with largest differences
    print("\nCountries where WID > PIP (top 10 by absolute difference):")
    top_wid_higher = plot_df.nlargest(10, 'difference')[['country', 'pip_mean', 'wid_mean', 'difference']]
    for idx, row in top_wid_higher.iterrows():
        print(f"  {row['country']}: PIP=${row['pip_mean']:.2f}, WID=${row['wid_mean']:.2f}, Diff=+${row['difference']:.2f}")

    print("\nCountries where PIP > WID (top 10 by absolute difference):")
    top_pip_higher = plot_df.nsmallest(10, 'difference')[['country', 'pip_mean', 'wid_mean', 'difference']]
    for idx, row in top_pip_higher.iterrows():
        print(f"  {row['country']}: PIP=${row['pip_mean']:.2f}, WID=${row['wid_mean']:.2f}, Diff=${row['difference']:.2f}")


def create_interactive_plot(pip_wid_comparison, output_file='outputs/pip_vs_wid_mean_income_interactive.html'):
    """
    Create interactive plotly scatter plot comparing PIP vs WID mean income by country.

    Args:
        pip_wid_comparison: DataFrame with 'country', 'pip_mean', 'wid_mean', 'wid_population'
        output_file: path to save the HTML file
    """
    # Remove any countries with missing data
    plot_df = pip_wid_comparison.dropna().copy()

    print(f"\nCreating interactive plot with {len(plot_df)} countries")

    # Identify the 10 most populous countries (by WID population)
    top10_countries = plot_df.nlargest(10, 'wid_population')

    print(f"\nTop 10 most populous countries:")
    for idx, row in top10_countries.iterrows():
        print(f"  {row['country']}: {row['wid_population']:,.0f}")

    # Separate data: top 10 most populous vs others
    top10_mask = plot_df['country'].isin(top10_countries['country'])
    other_countries = plot_df[~top10_mask]
    top10_data = plot_df[top10_mask]

    # Fit regression lines in log-log space
    X_log = np.log(plot_df[['pip_mean']].values)
    y_log = np.log(plot_df['wid_mean'].values)

    # Unweighted regression
    reg_unweighted = LinearRegression()
    reg_unweighted.fit(X_log, y_log)

    # Weighted regression
    reg_weighted = LinearRegression()
    weights = plot_df['wid_population'].values
    reg_weighted.fit(X_log, y_log, sample_weight=weights)

    # Create sorted data for regression lines
    X = plot_df['pip_mean'].values
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    X_log_sorted = np.log(X_sorted).reshape(-1, 1)

    y_log_unweighted = reg_unweighted.predict(X_log_sorted)
    y_unweighted = np.exp(y_log_unweighted)

    y_log_weighted = reg_weighted.predict(X_log_sorted)
    y_weighted = np.exp(y_log_weighted)

    # Create figure
    fig = go.Figure()

    # Add reference lines (1x, 2x, 3x, 5x, 7x, 10x, 20x)
    multipliers = [1, 2, 3, 5, 7, 10, 20]
    min_val = plot_df['pip_mean'].min() * 0.8
    max_val = plot_df['pip_mean'].max() * 1.2
    x_range = np.array([min_val, max_val])

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

        # Add text annotation on the line (in log space, position at ~70% along x-axis)
        label_x = np.exp(np.log(min_val) + 0.7 * (np.log(max_val) - np.log(min_val)))
        label_y = label_x * mult

        reference_line_annotations.append(
            dict(
                x=np.log10(label_x),
                y=np.log10(label_y),
                text=label_text,
                showarrow=False,
                font=dict(size=10, color=color if mult == 1 else 'gray'),
                bgcolor='rgba(255,255,255,0.7)',
                borderpad=2
            )
        )

    # Add regression lines
    fig.add_trace(go.Scatter(
        x=X_sorted,
        y=y_unweighted,
        mode='lines',
        line=dict(color='blue', width=2.5),
        name='Unweighted regression',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=X_sorted,
        y=y_weighted,
        mode='lines',
        line=dict(color='purple', width=2.5),
        name='Population-weighted regression',
        hoverinfo='skip'
    ))

    # Add scatter for other countries
    hover_text_other = []
    for idx, row in other_countries.iterrows():
        text = (f"<b>{row['country']}</b><br>"
                f"PIP: ${row['pip_mean']:.2f}/day (${row['pip_mean']*365:.0f}/year)<br>"
                f"WID: ${row['wid_mean']:.2f}/day (${row['wid_mean']*365:.0f}/year)<br>"
                f"WID/PIP ratio: {row['wid_mean']/row['pip_mean']:.2f}x")
        hover_text_other.append(text)

    fig.add_trace(go.Scatter(
        x=other_countries['pip_mean'],
        y=other_countries['wid_mean'],
        mode='markers',
        marker=dict(size=8, color='#2E86AB', opacity=0.6, line=dict(color='white', width=0.5)),
        name='Countries',
        text=hover_text_other,
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add scatter for top 10 countries
    hover_text_top10 = []
    for idx, row in top10_data.iterrows():
        text = (f"<b>{row['country']}</b> (Pop: {row['wid_population']/1e9:.2f}B)<br>"
                f"PIP: ${row['pip_mean']:.2f}/day (${row['pip_mean']*365:.0f}/year)<br>"
                f"WID: ${row['wid_mean']:.2f}/day (${row['wid_mean']*365:.0f}/year)<br>"
                f"WID/PIP ratio: {row['wid_mean']/row['pip_mean']:.2f}x")
        hover_text_top10.append(text)

    fig.add_trace(go.Scatter(
        x=top10_data['pip_mean'],
        y=top10_data['wid_mean'],
        mode='markers+text',
        marker=dict(size=15, color='#E63946', opacity=0.9, line=dict(color='black', width=1.5)),
        name='Top 10 most populous',
        text=top10_data['country'],
        textposition='top center',
        textfont=dict(size=11, color='black'),
        customdata=hover_text_top10,
        hovertemplate='%{customdata}<extra></extra>'
    ))

    # Update layout
    slope_unw = reg_unweighted.coef_[0]
    intercept_unw = reg_unweighted.intercept_
    slope_w = reg_weighted.coef_[0]
    intercept_w = reg_weighted.intercept_

    annotation_text = (f"Regression (log-log):<br>"
                      f"Unweighted: WID = {np.exp(intercept_unw):.2f} × PIP<sup>{slope_unw:.2f}</sup><br>"
                      f"Weighted: WID = {np.exp(intercept_w):.2f} × PIP<sup>{slope_w:.2f}</sup>")

    fig.update_layout(
        title=dict(
            text='Comparison of Mean Income by Country (per capita)<br>PIP vs WID (2023)',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='PIP Mean Income ($ per day, 2021 PPP)',
            type='log',
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title='WID Mean Income ($ per day, 2021 PPP)',
            type='log',
            gridcolor='lightgray',
            gridwidth=0.5,
            scaleanchor='x',
            scaleratio=1
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
    print(f"Interactive plot saved to: {output_file}")

    return fig


def main():
    """Main execution function."""
    print("="*70)
    print("PIP vs WID MEAN INCOME COMPARISON (both per capita)")
    print("="*70)

    # Load harmonized data
    print("\nLoading harmonized data...")
    df = pd.read_csv('modified/pip_wid_harmonized.csv')
    print(f"Loaded {len(df):,} rows")

    # Filter to only PIP and WID_per_capita for comparison (both are per capita measures)
    print("\nFiltering to PIP and WID_per_capita for comparison...")
    df_filtered = df[df['source'].isin(['PIP', 'WID_per_capita'])].copy()
    print(f"Filtered to {len(df_filtered):,} rows")

    # Calculate mean income for each country and source
    print("\nCalculating country mean incomes...")
    country_means = calculate_country_means(df_filtered)
    print(f"Calculated means for {len(country_means)} country-source combinations")

    # Pivot to get PIP and WID side by side (for mean income)
    pip_wid_comparison = country_means.pivot(
        index='country',
        columns='source',
        values='mean_income'
    ).reset_index()
    pip_wid_comparison.columns = ['country', 'PIP', 'WID_per_capita']
    # Rename for compatibility with existing code
    pip_wid_comparison = pip_wid_comparison.rename(columns={'PIP': 'pip_mean', 'WID_per_capita': 'wid_mean'})

    # Pivot to get population (we'll use WID population for weighting)
    country_pop = country_means.pivot(
        index='country',
        columns='source',
        values='population'
    ).reset_index()

    # Rename columns to match what we expect
    if 'PIP' in country_pop.columns:
        country_pop = country_pop.rename(columns={'PIP': 'pip_population'})
    if 'WID_per_capita' in country_pop.columns:
        country_pop = country_pop.rename(columns={'WID_per_capita': 'wid_population'})

    # Merge population into comparison dataframe
    pip_wid_comparison = pip_wid_comparison.merge(
        country_pop[['country', 'wid_population']],
        on='country',
        how='left'
    )

    print(f"\nCountries with PIP data: {pip_wid_comparison['pip_mean'].notna().sum()}")
    print(f"Countries with WID data: {pip_wid_comparison['wid_mean'].notna().sum()}")
    print(f"Countries with both: {pip_wid_comparison.dropna().shape[0]}")

    # Print summary statistics
    print_summary_statistics(pip_wid_comparison)

    # Create interactive plot
    print("\n" + "="*70)
    print("Creating interactive scatter plot...")
    create_interactive_plot(pip_wid_comparison)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
