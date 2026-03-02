"""
Shared Mean Log Deviation (MLD) decomposition calculation.

This module provides a single authoritative implementation of MLD decomposition
that is used consistently across all analysis scripts in the project.

MLD Decomposition Formula:
- Total MLD = Between-country MLD + Within-country MLD
- Between-country: Σ (pop_country / pop_total) × ln(global_mean / country_mean)
- Within-country: Σ (pop_country / pop_total) × MLD_country
  where MLD_country = Σ (pop_bin / pop_country) × ln(country_mean / bin_income)

Zero Income Handling:
The MLD formula requires ln(mean / income), which is undefined when income = 0.
We replace zero incomes with $0.01/day (1 cent) before calculation. This:
- Allows mathematical calculation to proceed
- Has minimal impact on results (zeros are rare and in small-population bins)
- Is more defensible than excluding people entirely
- Treats zero as "extremely low income" rather than missing data
"""

import pandas as pd
import numpy as np

# Minimum income floor to replace zeros (1 cent per day)
MIN_INCOME = 0.01


def calculate_mld_decomposition(df, income_col='average', pop_col='pop', country_col='country'):
    """
    Calculate Mean Log Deviation (MLD) decomposition into between and within components.

    Args:
        df: DataFrame with income, population, and country columns
        income_col: Name of column containing income values
        pop_col: Name of column containing population values
        country_col: Name of column containing country identifiers

    Returns:
        dict with keys:
            - total_mld: Total MLD across all countries
            - between_mld: Between-country component
            - within_mld: Within-country component
            - between_share: Between-country share (0-1)
            - within_share: Within-country share (0-1)
            - global_mean: Global mean income
            - total_pop: Total population
            - num_countries: Number of countries
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Replace zeros with minimum income floor
    zeros_before = (df[income_col] == 0).sum()
    df.loc[df[income_col] <= 0, income_col] = MIN_INCOME

    if zeros_before > 0:
        print(f"  Note: Replaced {zeros_before} zero-income rows with ${MIN_INCOME}/day")

    # Calculate global mean
    total_income = (df[income_col] * df[pop_col]).sum()
    total_pop = df[pop_col].sum()
    global_mean = total_income / total_pop

    # Calculate country-level statistics
    country_stats = df.groupby(country_col).apply(
        lambda x: pd.Series({
            'country_mean': (x[income_col] * x[pop_col]).sum() / x[pop_col].sum(),
            'country_pop': x[pop_col].sum()
        })
    ).reset_index()

    num_countries = len(country_stats)

    # Between-country MLD
    # MLD_between = Σ (pop_country / pop_total) × ln(global_mean / country_mean)
    between_mld = (
        (country_stats['country_pop'] / total_pop) *
        np.log(global_mean / country_stats['country_mean'])
    ).sum()

    # Within-country MLD for each country
    within_mlds = []
    for country in df[country_col].unique():
        country_df = df[df[country_col] == country].copy()

        # Calculate this country's mean and population
        country_mean = (country_df[income_col] * country_df[pop_col]).sum() / country_df[pop_col].sum()
        country_pop = country_df[pop_col].sum()

        # Calculate MLD within this country
        # MLD_country = Σ (pop_bin / pop_country) × ln(country_mean / bin_income)
        country_mld = (
            (country_df[pop_col] / country_pop) *
            np.log(country_mean / country_df[income_col])
        ).sum()

        within_mlds.append({
            'country': country,
            'within_mld': country_mld,
            'pop_weight': country_pop / total_pop
        })

    # Aggregate within-country MLDs weighted by population
    within_df = pd.DataFrame(within_mlds)
    within_mld = (within_df['within_mld'] * within_df['pop_weight']).sum()

    # Total MLD = Between + Within
    total_mld = between_mld + within_mld

    return {
        'total_mld': total_mld,
        'between_mld': between_mld,
        'within_mld': within_mld,
        'between_share': between_mld / total_mld if total_mld > 0 else 0,
        'within_share': within_mld / total_mld if total_mld > 0 else 0,
        'global_mean': global_mean,
        'total_pop': total_pop,
        'num_countries': num_countries
    }


def calculate_mld_for_multiple_sources(df, source_col='source'):
    """
    Calculate MLD decomposition for multiple data sources in a single DataFrame.

    Useful for comparing PIP vs WID, or different income concepts side-by-side.

    Args:
        df: DataFrame with 'source', 'country', 'average', 'pop' columns
        source_col: Name of column distinguishing different data sources

    Returns:
        dict mapping source names to decomposition results
    """
    results = {}

    for source in df[source_col].unique():
        source_df = df[df[source_col] == source]
        results[source] = calculate_mld_decomposition(source_df)

    return results
