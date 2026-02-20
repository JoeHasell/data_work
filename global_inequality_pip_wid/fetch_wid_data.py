"""
Robust WID data fetching with country-by-country API calls.

This script orchestrates data fetching from WID API by:
1. Fetching PPP and population data (fast, all countries at once)
2. Fetching percentile data country-by-country (resilient to API failures)
3. Tracking progress and allowing resume if interrupted
4. Combining all country files at the end

Usage:
    python fetch_wid_data.py              # Start fresh fetch
    python fetch_wid_data.py --resume     # Resume from last saved state
    python fetch_wid_data.py --country US # Fetch single country only
"""

import os
import sys
import json
import time
import subprocess
import argparse
import pandas as pd
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUTS_DIR = "inputs"
TEMP_DIR = os.path.join(INPUTS_DIR, "temp_country_data")
STATE_FILE = os.path.join(INPUTS_DIR, "fetch_progress.json")
COUNTRY_MAPPING_FILE = os.path.join(INPUTS_DIR, "country_mapping.csv")

STATA_PATH = "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se"
TARGET_YEAR = 2023

# Output files
OUTPUT_PPP = os.path.join(INPUTS_DIR, "WID_ppp.csv")
OUTPUT_POPULATION = os.path.join(INPUTS_DIR, "WID_aggregate_population.csv")
OUTPUT_PERCENTILES = os.path.join(INPUTS_DIR, "WID_percentiles.csv")


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg):
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {msg}")


def log_success(msg):
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}")


def log_warning(msg):
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {msg}")


def log_error(msg):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {msg}")


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state():
    """Load progress state from JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'completed_countries': [],
        'failed_countries': [],
        'last_updated': None,
        'ppp_done': False,
        'population_done': False
    }


def save_state(state):
    """Save progress state to JSON file."""
    state['last_updated'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def clear_state():
    """Remove state file to start fresh."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        log_info("Cleared previous progress state")


# =============================================================================
# COUNTRY MANAGEMENT
# =============================================================================

def load_country_codes():
    """Load country codes from mapping file."""
    df = pd.read_csv(COUNTRY_MAPPING_FILE)
    # Filter out countries with empty PIP country name (not in PIP dataset)
    df = df[df['PIP country name'].notna() & (df['PIP country name'] != '')]
    codes = df['country'].tolist()
    log_info(f"Loaded {len(codes)} country codes from {COUNTRY_MAPPING_FILE}")
    return codes


def get_countries_to_fetch(all_countries, state, single_country=None):
    """Determine which countries need to be fetched."""
    if single_country:
        if single_country not in all_countries:
            log_error(f"Country code '{single_country}' not found in mapping file")
            return []
        return [single_country]

    completed = set(state['completed_countries'])
    remaining = [c for c in all_countries if c not in completed]

    if remaining:
        log_info(f"Progress: {len(completed)}/{len(all_countries)} countries completed")
        log_info(f"Remaining: {len(remaining)} countries")

    return remaining


# =============================================================================
# STATA EXECUTION
# =============================================================================

def run_stata_script(do_file, timeout=600):
    """
    Run a Stata .do file and return success status.

    Args:
        do_file: Path to .do file (can be absolute or relative)
        timeout: Timeout in seconds (default 10 minutes)

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(STATA_PATH):
        log_error(f"Stata not found at {STATA_PATH}")
        return False

    # Check if file exists
    if not os.path.exists(do_file):
        log_error(f"Do file not found: {do_file}")
        return False

    # Get just the filename for Stata (since we run from INPUTS_DIR)
    do_filename = os.path.basename(do_file)

    try:
        result = subprocess.run(
            [STATA_PATH, "-b", "do", do_filename],
            cwd=INPUTS_DIR,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Check for specific WID API errors in log
        log_file = do_file.replace('.do', '.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
                if 'could not access the online WID.world database' in log_content:
                    log_error("WID API connection error")
                    return False
                if 'r(677)' in log_content or 'r(603)' in log_content:
                    return False

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        log_error(f"Stata execution timed out after {timeout} seconds")
        return False
    except Exception as e:
        log_error(f"Error running Stata: {e}")
        return False


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_ppp_data(state, force=False):
    """Fetch PPP conversion factors (fast, all countries at once)."""
    if state['ppp_done'] and not force:
        log_info("PPP data already fetched (use --force to re-fetch)")
        return True

    log_info(f"Fetching PPP data for {TARGET_YEAR}...")

    do_file = os.path.join(INPUTS_DIR, "fetch_ppp.do")

    if run_stata_script(do_file, timeout=300):
        if os.path.exists(OUTPUT_PPP):
            state['ppp_done'] = True
            save_state(state)
            log_success(f"PPP data saved to {OUTPUT_PPP}")
            return True

    log_error("Failed to fetch PPP data")
    return False


def fetch_population_data(state, force=False):
    """Fetch population data (fast, all countries at once)."""
    if state['population_done'] and not force:
        log_info("Population data already fetched (use --force to re-fetch)")
        return True

    log_info(f"Fetching population data for {TARGET_YEAR}...")

    do_file = os.path.join(INPUTS_DIR, "fetch_population.do")

    if run_stata_script(do_file, timeout=300):
        if os.path.exists(OUTPUT_POPULATION):
            state['population_done'] = True
            save_state(state)
            log_success(f"Population data saved to {OUTPUT_POPULATION}")
            return True

    log_error("Failed to fetch population data")
    return False


def fetch_country_percentiles(country_code, retry_count=3):
    """
    Fetch percentile data for a single country.

    Args:
        country_code: 2-letter country code
        retry_count: Number of retries on failure

    Returns:
        bool: True if successful, False otherwise
    """
    output_file = os.path.join(TEMP_DIR, f"{country_code}_percentiles.csv")

    # Check if already exists
    if os.path.exists(output_file):
        return True

    # Create temporary .do file with country code
    # Note: Stata runs from INPUTS_DIR, so use relative path
    temp_dir_rel = os.path.basename(TEMP_DIR)  # Just "temp_country_data"

    do_content = f"""
* Fetch percentiles for {country_code}
local target_year = {TARGET_YEAR}
local country_code = "{country_code}"
local outdir = "{temp_dir_rel}"

* Percentile list
local perc_list ///
p0p1 p1p2 p2p3 p3p4 p4p5 p5p6 p6p7 p7p8 p8p9 p9p10 ///
p10p11 p11p12 p12p13 p13p14 p14p15 p15p16 p16p17 p17p18 p18p19 p19p20 ///
p20p21 p21p22 p22p23 p23p24 p24p25 p25p26 p26p27 p27p28 p28p29 p29p30 ///
p30p31 p31p32 p32p33 p33p34 p34p35 p35p36 p36p37 p37p38 p38p39 p39p40 ///
p40p41 p41p42 p42p43 p43p44 p44p45 p45p46 p46p47 p47p48 p48p49 p49p50 ///
p50p51 p51p52 p52p53 p53p54 p54p55 p55p56 p56p57 p57p58 p58p59 p59p60 ///
p60p61 p61p62 p62p63 p63p64 p64p65 p65p66 p66p67 p67p68 p68p69 p69p70 ///
p70p71 p71p72 p72p73 p73p74 p74p75 p75p76 p76p77 p77p78 p78p79 p79p80 ///
p80p81 p81p82 p82p83 p83p84 p84p85 p85p86 p86p87 p87p88 p88p89 p89p90 ///
p90p91 p91p92 p92p93 p93p94 p94p95 p95p96 p96p97 p97p98 p98p99 ///
p99p99.1 p99.1p99.2 p99.2p99.3 p99.3p99.4 p99.4p99.5 ///
p99.5p99.6 p99.6p99.7 p99.7p99.8 p99.8p99.9 p99.9p100

* Download percentile data for this country
* Fetch two income concepts:
*   - Pre-tax national income (aptinc, sptinc)
*   - Post-tax disposable income (adiinc, sdiinc)
wid, indicators(aptinc sptinc adiinc sdiinc) ///
    areas(`country_code')    ///
    years(`target_year')     ///
    perc(`perc_list')        ///
    ages(992)                ///
    population(j)            ///
    clear

* Keep relevant variables
keep country year percentile variable value

* Strip the "992j" suffix to get indicator name
gen indicator = substr(variable, 1, 6)
drop variable

* Reshape wide to get columns for each indicator
reshape wide value, i(country year percentile) j(indicator) string

* Rename with descriptive suffixes (only if variable exists)
* Pre-tax national income
capture rename valueaptinc avg_pretax
capture rename valuesptinc share_pretax

* Post-tax disposable income
capture rename valueadiinc avg_posttax
capture rename valuesdiinc share_posttax

* Parse percentile bounds
gen str10 p_clean = substr(percentile, 2, .)
split p_clean, parse("p") gen(p_)
destring p_1 p_2, replace
gen double p_low = p_1 / 100
gen double p_high = p_2 / 100
drop p_clean p_1 p_2

* Keep all income variables that exist plus metadata
* Can't use explicit list because not all countries have all income concepts
keep country percentile year avg_* share_* p_low p_high

* Export
export delimited using "`outdir'/`country_code'_percentiles.csv", replace delim(",")
"""

    temp_do = os.path.join(INPUTS_DIR, f"temp_fetch_{country_code}.do")

    for attempt in range(retry_count):
        try:
            # Write temporary .do file
            with open(temp_do, 'w') as f:
                f.write(do_content)

            # Run Stata
            success = run_stata_script(temp_do, timeout=180)  # 3 minute timeout per country

            # Clean up temp files
            if os.path.exists(temp_do):
                os.remove(temp_do)
            temp_log = temp_do.replace('.do', '.log')
            if os.path.exists(temp_log):
                os.remove(temp_log)

            if success and os.path.exists(output_file):
                return True

            if attempt < retry_count - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                log_warning(f"Retry {attempt + 1}/{retry_count} for {country_code} after {wait_time}s...")
                time.sleep(wait_time)

        except Exception as e:
            log_error(f"Error fetching {country_code}: {e}")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)

    return False


def combine_country_files(country_codes):
    """Combine individual country CSV files into single percentiles file."""
    log_info(f"Combining {len(country_codes)} country files...")

    dfs = []
    missing = []

    for code in country_codes:
        file_path = os.path.join(TEMP_DIR, f"{code}_percentiles.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                log_warning(f"Error reading {code}: {e}")
                missing.append(code)
        else:
            missing.append(code)

    if missing:
        # Convert to strings in case there are NaN values
        missing_str = [str(m) for m in missing]
        log_warning(f"Missing data for {len(missing)} countries: {', '.join(missing_str[:10])}...")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(OUTPUT_PERCENTILES, index=False)
        log_success(f"Combined {len(dfs)} countries into {OUTPUT_PERCENTILES}")
        log_info(f"Total rows: {len(combined):,}")
        return True

    log_error("No country files to combine")
    return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch WID data with robust country-by-country approach"
    )
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last saved progress')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch PPP and population even if already done')
    parser.add_argument('--country', type=str,
                        help='Fetch data for single country only (2-letter code)')
    parser.add_argument('--combine-only', action='store_true',
                        help='Skip fetching, just combine existing country files')

    args = parser.parse_args()

    # Setup
    os.makedirs(TEMP_DIR, exist_ok=True)

    print("=" * 80)
    print(f"{Colors.BOLD}WID Data Fetching - Robust Country-by-Country Approach{Colors.ENDC}")
    print("=" * 80)

    # Load state
    if args.resume:
        state = load_state()
        log_info("Resuming from previous progress")
    else:
        if not args.combine_only and not args.country:
            clear_state()
        state = load_state()

    # Load country codes
    all_countries = load_country_codes()

    # Combine only mode
    if args.combine_only:
        log_info("Combine-only mode: skipping fetch, combining existing files")
        success = combine_country_files(all_countries)
        return 0 if success else 1

    # Step 1: Fetch PPP data (fast)
    if not args.country:
        if not fetch_ppp_data(state, force=args.force):
            log_error("Failed to fetch PPP data - aborting")
            return 1

        # Step 2: Fetch population data (fast)
        if not fetch_population_data(state, force=args.force):
            log_error("Failed to fetch population data - aborting")
            return 1

    # Step 3: Fetch percentile data country-by-country
    countries_to_fetch = get_countries_to_fetch(all_countries, state, args.country)

    if not countries_to_fetch:
        log_success("All countries already fetched!")
    else:
        log_info(f"Fetching percentile data for {len(countries_to_fetch)} countries...")

        start_time = time.time()
        successful = 0
        failed = 0

        for i, country_code in enumerate(countries_to_fetch, 1):
            log_info(f"[{i}/{len(countries_to_fetch)}] Fetching {country_code}...")

            if fetch_country_percentiles(country_code):
                successful += 1
                state['completed_countries'].append(country_code)
                save_state(state)
                log_success(f"{country_code} completed")
            else:
                failed += 1
                state['failed_countries'].append(country_code)
                save_state(state)
                log_error(f"{country_code} failed")

            # Progress update every 10 countries
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = len(countries_to_fetch) - i
                eta = remaining / rate if rate > 0 else 0
                log_info(f"Progress: {i}/{len(countries_to_fetch)} | "
                        f"Success: {successful} | Failed: {failed} | "
                        f"ETA: {eta/60:.1f} min")

        elapsed = time.time() - start_time
        log_info(f"Fetching completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        log_info(f"Successful: {successful}/{len(countries_to_fetch)}")
        if failed > 0:
            log_warning(f"Failed: {failed}/{len(countries_to_fetch)}")

    # Step 4: Combine country files (unless single country mode)
    if not args.country:
        if combine_country_files(all_countries):
            log_success("All data fetching complete!")
            log_info(f"Output files:")
            log_info(f"  - {OUTPUT_PPP}")
            log_info(f"  - {OUTPUT_POPULATION}")
            log_info(f"  - {OUTPUT_PERCENTILES}")
        else:
            log_error("Failed to combine country files")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
