# Global Inequality Analysis: PIP vs WID

Analysis of global income inequality using data from two sources: World Bank's Poverty and Inequality Platform (PIP) and the World Inequality Database (WID).

## Research Question

Why do PIP and WID give different answers about the composition of global inequality?

- **PIP**: 69% of global inequality is between countries
- **WID**: 64% of global inequality is within countries

This project investigates what drives this 33.5 percentage point gap using Mean Log Deviation (MLD) decomposition and counterfactual analyses.

## Quick Start

### Prerequisites

1. **Python 3.7+** with pip
2. **Stata** (for fetching WID data)
   - Update `STATA_PATH` in `config.py` if not using default location

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Run all steps from data fetch to final analysis
python run_all.py

# Skip Stata data fetch (use existing data)
python run_all.py --skip-stata

# Run specific steps only
python run_all.py --step 3-8  # Skip data fetching, run analysis only

# List all available steps
python run_all.py --list
```

## Project Structure

```
.
├── run_all.py              # Main pipeline orchestrator
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # Detailed session history
│
├── inputs/                # Raw data (from WID API)
│   ├── WID_ppp.csv
│   ├── WID_percentiles.csv
│   ├── WID_aggregate_population.csv
│   ├── country_mapping.csv
│   └── fetch_percentiles_aggs_and_pop.do  # Stata script
│
├── modified/              # Intermediate processed data
│   ├── WID_percentiles_with_per_capita.csv
│   └── pip_wid_harmonized.csv
│
└── outputs/               # Analysis results and visualizations
    ├── inequality_decomposition.png
    ├── pip_vs_wid_mean_income_interactive.html
    ├── pip_vs_wid_country_mlds.html
    ├── counterfactual_decomposition.png
    └── shape_vs_mean_decomposition.png
```

## Pipeline Steps

1. **Fetch WID data** (`fetch_wid_data.py` - robust country-by-country approach)
   - Downloads data from WID API for year 2023
   - Fetches PPP and population data (fast)
   - Fetches percentile data country-by-country for resilience
   - Tracks progress and allows resume if interrupted
   - Creates: WID_ppp.csv, WID_percentiles.csv, WID_aggregate_population.csv
   - Time: ~20-40 minutes (depending on API speed)
   - See [WID_FETCHING.md](WID_FETCHING.md) for details

2. **Create per capita adjustments** (`create_wid_per_capita.py`)
   - Converts WID data from per-adult to per-capita basis
   - Creates: WID_percentiles_with_per_capita.csv

3. **Harmonize data** (`harmonize_percentiles.py`)
   - Downloads PIP data from URL
   - Aggregates PIP from 1000 bins to 101 bins matching WID structure
   - Merges PIP and WID data
   - Creates: pip_wid_harmonized.csv

4. **Analyze inequality** (`analyze_inequality.py`)
   - Calculates MLD decomposition (between vs within country)
   - Creates: inequality_decomposition.png

5. **Comparison plots** (`comparison_plots.py`)
   - Compares PIP vs WID mean incomes across countries
   - Creates: pip_vs_wid_mean_income_interactive.html

6. **Country MLDs** (`compare_country_mlds.py`)
   - Compares within-country inequality for each country
   - Creates: pip_vs_wid_country_mlds.html

7. **Counterfactual analysis** (`counterfactual_analysis.py`)
   - Tests impact of swapping inequality distributions between countries
   - Creates: counterfactual_decomposition.png

8. **Shape vs mean analysis** (`shape_vs_mean_analysis.py`)
   - Isolates whether gap is due to mean incomes or distribution shapes
   - Creates: shape_vs_mean_decomposition.png

## Key Findings

### Main Result

**87% of the PIP-WID gap is driven by distribution shapes** (different within-country inequality patterns), not different mean income levels.

### Data Differences

- **PIP**: Per capita income (total population)
- **WID**: Per adult income (adult population only)
- **WID reports ~2.78x higher mean incomes** than PIP (median: 2.47x)
- **WID shows ~3x higher within-country inequality** in 100% of countries

### MLD Decomposition (2023)

| Source | Total MLD | Between-Country | Within-Country |
|--------|-----------|-----------------|----------------|
| PIP (per capita) | 0.69 | 69.2% | 30.8% |
| WID (per adult) | 1.00 | 28.3% | 71.7% |
| WID (per capita) | 1.10 | 35.7% | 64.3% |

## Configuration

Edit `config.py` to customize:
- Target year
- File paths
- Stata executable location
- Analysis parameters

## Running Individual Scripts

Each script can also be run independently:

```bash
python create_wid_per_capita.py
python harmonize_percentiles.py
python analyze_inequality.py
# etc.
```

## Data Sources

- **PIP**: [OWID Data Catalog](https://catalog.ourworldindata.org/garden/wb/2025-10-13/thousand_bins_distribution/thousand_bins_distribution.feather)
- **WID**: Fetched via Stata `wid` command from [WID.world](https://wid.world/)

## Notes

- The Stata data fetch can be slow (10-20 minutes) as it downloads ~28,000 rows from the WID API
- Use `--skip-stata` flag if you already have the WID data files
- All visualizations are interactive HTML files (can be opened in browser)
- See `CLAUDE.md` for detailed session history and development notes

## Troubleshooting

**Stata not found:**
- Update `STATA_PATH` in `config.py` to your Stata installation

**Missing Python packages:**
```bash
pip install -r requirements.txt
```

**Data fetch timeout:**
- The WID API can be slow; the timeout is set to 20 minutes
- If it fails, try running again or use `--skip-stata` with existing data

## Documentation

- **README.md** (this file) - Project overview and quick start guide
- **[WID_FETCHING.md](WID_FETCHING.md)** - Technical documentation for the robust WID data fetching system
- **.claude.md** - Development session history and detailed implementation notes (hidden file)

## License

Research project - contact author for licensing details.
