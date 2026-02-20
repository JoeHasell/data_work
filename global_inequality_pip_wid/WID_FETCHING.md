# WID Data Fetching - Robust Approach

## Overview

The WID API can be unreliable when fetching large amounts of data (all countries × all percentiles in one call). This project uses a **robust country-by-country approach** that:

1. Fetches PPP and population data separately (fast, rarely fails)
2. Fetches percentile data one country at a time
3. Tracks progress and allows resuming if interrupted
4. Stores each country's data separately until all are fetched
5. Combines country files at the end

## Architecture

### Files

- **`fetch_wid_data.py`** - Python orchestrator that manages the entire process
- **`inputs/fetch_ppp.do`** - Stata script to fetch PPP conversion factors
- **`inputs/fetch_population.do`** - Stata script to fetch population data
- **`inputs/temp_country_data/`** - Temporary directory storing individual country CSV files
- **`inputs/fetch_progress.json`** - Progress tracking (which countries completed/failed)

### How It Works

```
1. Fetch PPP data         → inputs/WID_ppp.csv
   ↓ (all countries, fast)

2. Fetch population data  → inputs/WID_aggregate_population.csv
   ↓ (all countries, fast)

3. For each country:
   ├─ Generate temp .do file with country code
   ├─ Call Stata to fetch that country's percentiles
   ├─ Save to inputs/temp_country_data/{country}_percentiles.csv
   ├─ Update progress in fetch_progress.json
   └─ Retry up to 3 times if API fails

4. Combine all country files → inputs/WID_percentiles.csv
   ↓ (merge all CSV files)

5. Clean up temporary files
```

## Usage

### Basic Usage

```bash
# Fetch all WID data (fresh start)
python fetch_wid_data.py

# Resume after interruption
python fetch_wid_data.py --resume

# Re-fetch PPP and population even if already done
python fetch_wid_data.py --force

# Fetch single country for testing
python fetch_wid_data.py --country US

# Combine existing country files without fetching
python fetch_wid_data.py --combine-only
```

### Within Pipeline

The main pipeline automatically uses this approach:

```bash
# Full pipeline with robust WID fetching
python run_all.py

# Skip WID fetching (use existing data)
python run_all.py --skip-stata
```

## Progress Tracking

Progress is saved in `inputs/fetch_progress.json`:

```json
{
  "completed_countries": ["AD", "AE", "AF", ...],
  "failed_countries": ["XX", "YY"],
  "last_updated": "2026-02-20T16:30:00",
  "ppp_done": true,
  "population_done": true
}
```

If the script is interrupted (Ctrl+C, connection loss, etc.), you can resume exactly where you left off:

```bash
python fetch_wid_data.py --resume
```

## Benefits

### Compared to Old Approach

**Old approach (all at once):**
- ❌ Single API call for all countries × all percentiles (~28,000 rows)
- ❌ If API fails after 15 minutes, lose all progress
- ❌ Can't track which countries succeeded
- ❌ Must restart from scratch

**New approach (country-by-country):**
- ✅ Smaller API calls (109 percentiles per country)
- ✅ Progress saved after each country
- ✅ Can resume from interruption
- ✅ Retry failed countries individually
- ✅ Clear progress tracking and ETA
- ✅ Can test with single country first

### Performance

- **PPP data**: ~30 seconds (270 countries)
- **Population data**: ~30 seconds (280+ countries)
- **Percentile data**: ~20-40 minutes (210+ countries, ~5-10 seconds per country)
- **Total**: ~20-45 minutes depending on API speed

## Error Handling

The script handles several types of failures:

1. **API connection errors** - Retries up to 3 times with exponential backoff
2. **Timeout errors** - 3-minute timeout per country
3. **Partial data** - Tracks which countries failed, can re-run just those
4. **Interruption** - Saves progress, can resume

## Troubleshooting

### Script stuck on a country

The timeout is 3 minutes per country. If it's taking longer, the API might be slow. The script will automatically retry.

### Many countries failing

Check your internet connection. The WID API may also be experiencing issues. Wait and try again later.

### Resume not working

Check `inputs/fetch_progress.json`. You can manually edit it to remove countries from `completed_countries` if you want to re-fetch them.

### Missing countries in final output

Check `inputs/fetch_progress.json` for `failed_countries`. You can re-run just those:

```bash
python fetch_wid_data.py --country XX
```

### Clean start

Remove the progress file to start fresh:

```bash
rm inputs/fetch_progress.json
rm -rf inputs/temp_country_data/
python fetch_wid_data.py
```

## Country Codes

Country codes come from `inputs/country_mapping.csv`. Only countries with a corresponding PIP country name are fetched (typically ~210-220 countries).

Examples:
- `US` - United States
- `FR` - France
- `CN` - China
- `IN` - India
- `BR` - Brazil

See the full list in `country_mapping.csv`.

## Technical Details

### Stata Scripts

Each country fetch generates a temporary .do file like:

```stata
wid, indicators(aptinc sptinc adiinc sdiinc) ///
    areas(US)                 ///
    years(2023)               ///
    perc(p0p1 p1p2 ... p99.9p100) ///
    ages(992)                 ///
    population(j)             ///
    clear
```

This fetches **two income concepts**:
- **Pre-tax national income**: `aptinc` (average), `sptinc` (share)
- **Post-tax disposable income**: `adiinc` (average), `sdiinc` (share)

For all 109 percentiles, for adults (age 992), using equal-split adults (population j).

**Note:** Country coverage varies by income concept. Some countries only have pre-tax data, others have both pre-tax and post-tax.

### Output Format

Each country file has columns:
- `country` - 2-letter code
- `percentile` - e.g., "p0p1", "p99p100"
- `year` - 2023
- `avg_pretax` - average pre-tax income (international dollars, annual)
- `share_pretax` - pre-tax income share (fraction)
- `avg_posttax` - average post-tax income (international dollars, annual)
- `share_posttax` - post-tax income share (fraction)
- `p_low` - percentile lower bound (0-1)
- `p_high` - percentile upper bound (0-1)

**Note:** Not all countries have data for both income concepts. Some only have pre-tax data. Missing values will be empty/null.

The final combined file merges all countries into a single CSV.
