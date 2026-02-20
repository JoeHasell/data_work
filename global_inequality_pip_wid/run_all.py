"""
Main orchestrator script for global inequality analysis pipeline.

This script runs the complete analysis pipeline from data fetching to final outputs.

Usage:
    python run_all.py                  # Run complete pipeline
    python run_all.py --step 1         # Run only step 1
    python run_all.py --step 2-5       # Run steps 2 through 5
    python run_all.py --skip-stata     # Skip Stata data fetch (use existing data)

Steps:
    1. Fetch WID data (Stata)
    2. Create per capita adjustments
    3. Harmonize PIP and WID data
    4. Analyze inequality decomposition
    5. Create comparison plots
    6. Compare country-level MLDs
    7. Run counterfactual analysis
    8. Run shape vs mean analysis
"""

import sys
import os
import subprocess
import argparse
import time
from datetime import datetime

# Import configuration
import config


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def log_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
    print("=" * 80)


def log_step(step_num, message):
    """Print a formatted step message."""
    print(f"\n{Colors.OKBLUE}[STEP {step_num}] {message}{Colors.ENDC}")


def log_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def log_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def log_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def log_info(message):
    """Print an info message."""
    print(f"  {message}")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_1_fetch_wid_data():
    """
    Step 1: Fetch WID data using robust country-by-country approach.

    Creates:
        - WID_ppp.csv
        - WID_percentiles.csv
        - WID_aggregate_population.csv

    Uses fetch_wid_data.py which fetches data country-by-country
    for resilience to API failures.
    """
    log_step(1, "Fetching WID data from API (robust country-by-country)")

    # Check if Stata exists
    if not os.path.exists(config.STATA_PATH):
        log_error(f"Stata not found at {config.STATA_PATH}")
        log_info("Please update STATA_PATH in config.py")
        return False

    log_info("Using robust fetching approach (country-by-country)")
    log_info(f"This may take 20-40 minutes depending on API speed...")
    log_info("Progress is saved - you can Ctrl+C and resume with --resume flag")

    start_time = time.time()

    # Run fetch_wid_data.py
    try:
        result = subprocess.run(
            [sys.executable, "fetch_wid_data.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            log_error("WID data fetching failed")
            if result.stderr:
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        log_error("Data fetching timed out (>1 hour)")
        return False
    except KeyboardInterrupt:
        log_warning("Data fetching interrupted - progress has been saved")
        log_info("Resume with: python fetch_wid_data.py --resume")
        return False
    except Exception as e:
        log_error(f"Error running fetch_wid_data.py: {e}")
        return False

    elapsed = time.time() - start_time

    # Verify outputs exist
    required_files = [
        config.WID_PPP_FILE,
        config.WID_PERCENTILES_FILE,
        config.WID_POPULATION_FILE
    ]

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        log_error("Some output files were not created:")
        for f in missing:
            log_info(f"  Missing: {os.path.basename(f)}")
        return False

    log_success(f"WID data fetched successfully ({elapsed:.1f} seconds / {elapsed/60:.1f} minutes)")
    for f in required_files:
        size = os.path.getsize(f) / 1024  # KB
        log_info(f"  Created: {os.path.basename(f)} ({size:.1f} KB)")

    return True


def step_2_create_per_capita():
    """
    Step 2: Create WID data with per capita adjustments.

    Reads: WID_percentiles.csv, WID_aggregate_population.csv
    Creates: WID_percentiles_with_per_capita.csv
    """
    log_step(2, "Creating per capita adjustments")

    log_info("Running: create_wid_per_capita.py")

    try:
        import create_wid_per_capita
        create_wid_per_capita.main()

        if not os.path.exists(config.WID_WITH_PER_CAPITA_FILE):
            log_error("Output file was not created")
            return False

        log_success("Per capita adjustments created successfully")
        return True

    except Exception as e:
        log_error(f"Error in create_wid_per_capita.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_3_harmonize_data():
    """
    Step 3: Harmonize PIP and WID percentile data structures.

    Reads: PIP data (from URL), WID_percentiles_with_per_capita.csv, WID_ppp.csv
    Creates: pip_wid_harmonized.csv
    """
    log_step(3, "Harmonizing PIP and WID data")

    log_info("Running: harmonize_percentiles.py")
    log_info("This will download PIP data from URL...")

    try:
        import harmonize_percentiles
        harmonize_percentiles.main()

        if not os.path.exists(config.HARMONIZED_FILE):
            log_error("Harmonized file was not created")
            return False

        log_success("Data harmonization completed successfully")
        return True

    except Exception as e:
        log_error(f"Error in harmonize_percentiles.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_4_analyze_inequality():
    """
    Step 4: Calculate MLD inequality decomposition.

    Reads: pip_wid_harmonized.csv
    Creates: inequality_decomposition.png
    """
    log_step(4, "Analyzing inequality decomposition (MLD)")

    log_info("Running: analyze_inequality.py")

    try:
        import analyze_inequality
        analyze_inequality.main()

        if not os.path.exists(config.INEQUALITY_DECOMPOSITION_CHART):
            log_error("Inequality decomposition chart was not created")
            return False

        log_success("Inequality analysis completed successfully")
        return True

    except Exception as e:
        log_error(f"Error in analyze_inequality.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_5_comparison_plots():
    """
    Step 5: Create PIP vs WID mean income comparison plots.

    Reads: pip_wid_harmonized.csv
    Creates: pip_vs_wid_mean_income_interactive.html
    """
    log_step(5, "Creating mean income comparison plots")

    log_info("Running: comparison_plots.py")

    try:
        import comparison_plots
        comparison_plots.main()

        if not os.path.exists(config.MEAN_INCOME_PLOT):
            log_error("Mean income plot was not created")
            return False

        log_success("Comparison plots created successfully")
        return True

    except Exception as e:
        log_error(f"Error in comparison_plots.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_6_country_mlds():
    """
    Step 6: Compare country-level within-country MLDs.

    Reads: pip_wid_harmonized.csv
    Creates: pip_vs_wid_country_mlds.html
    """
    log_step(6, "Comparing country-level MLDs")

    log_info("Running: compare_country_mlds.py")

    try:
        import compare_country_mlds
        compare_country_mlds.main()

        if not os.path.exists(config.COUNTRY_MLD_PLOT):
            log_error("Country MLD plot was not created")
            return False

        log_success("Country MLD comparison completed successfully")
        return True

    except Exception as e:
        log_error(f"Error in compare_country_mlds.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_7_counterfactual_analysis():
    """
    Step 7: Run counterfactual analysis.

    Reads: pip_wid_harmonized.csv
    Creates: counterfactual_decomposition.png
    """
    log_step(7, "Running counterfactual analysis")

    log_info("Running: counterfactual_analysis.py")

    try:
        import counterfactual_analysis
        counterfactual_analysis.main()

        if not os.path.exists(config.COUNTERFACTUAL_CHART):
            log_error("Counterfactual chart was not created")
            return False

        log_success("Counterfactual analysis completed successfully")
        return True

    except Exception as e:
        log_error(f"Error in counterfactual_analysis.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_8_shape_vs_mean():
    """
    Step 8: Analyze shape vs mean income contributions.

    Reads: pip_wid_harmonized.csv
    Creates: shape_vs_mean_decomposition.png
    """
    log_step(8, "Analyzing shape vs mean contributions")

    log_info("Running: shape_vs_mean_analysis.py")

    try:
        import shape_vs_mean_analysis
        shape_vs_mean_analysis.main()

        if not os.path.exists(config.SHAPE_VS_MEAN_CHART):
            log_error("Shape vs mean chart was not created")
            return False

        log_success("Shape vs mean analysis completed successfully")
        return True

    except Exception as e:
        log_error(f"Error in shape_vs_mean_analysis.py: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================

PIPELINE_STEPS = [
    ("Fetch WID data (Stata)", step_1_fetch_wid_data),
    ("Create per capita adjustments", step_2_create_per_capita),
    ("Harmonize PIP and WID data", step_3_harmonize_data),
    ("Analyze inequality decomposition", step_4_analyze_inequality),
    ("Create comparison plots", step_5_comparison_plots),
    ("Compare country MLDs", step_6_country_mlds),
    ("Run counterfactual analysis", step_7_counterfactual_analysis),
    ("Analyze shape vs mean", step_8_shape_vs_mean),
]


def run_pipeline(start_step=1, end_step=None, skip_stata=False):
    """
    Run the complete analysis pipeline.

    Args:
        start_step: First step to run (1-indexed)
        end_step: Last step to run (1-indexed, None = run to end)
        skip_stata: Skip step 1 (Stata data fetch)

    Returns:
        bool: True if all steps succeeded, False otherwise
    """
    if end_step is None:
        end_step = len(PIPELINE_STEPS)

    # Adjust for skip_stata
    if skip_stata and start_step == 1:
        start_step = 2
        log_warning("Skipping Stata data fetch (using existing WID data)")

    log_header(f"GLOBAL INEQUALITY ANALYSIS PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"Running steps {start_step} to {end_step}")
    log_info(f"Target year: {config.TARGET_YEAR}")

    overall_start = time.time()
    failed_steps = []

    for i in range(start_step - 1, end_step):
        step_name, step_func = PIPELINE_STEPS[i]
        step_num = i + 1

        try:
            success = step_func()

            if not success:
                failed_steps.append((step_num, step_name))
                log_error(f"Step {step_num} failed: {step_name}")

                # Ask if user wants to continue (only in interactive mode)
                if sys.stdin.isatty():
                    print(f"\n{Colors.WARNING}Step {step_num} failed. Continue anyway? (y/n): {Colors.ENDC}", end='')
                    response = input().strip().lower()

                    if response != 'y':
                        log_error("Pipeline aborted by user")
                        break
                    else:
                        log_warning("Continuing despite failure...")
                else:
                    # Non-interactive mode - abort on failure
                    log_error("Pipeline aborted due to failure (non-interactive mode)")
                    break

        except KeyboardInterrupt:
            log_error("\nPipeline interrupted by user")
            return False
        except Exception as e:
            log_error(f"Unexpected error in step {step_num}: {e}")
            import traceback
            traceback.print_exc()
            failed_steps.append((step_num, step_name))
            break

    overall_elapsed = time.time() - overall_start

    # Summary
    log_header("PIPELINE SUMMARY")

    if failed_steps:
        log_error(f"Pipeline completed with {len(failed_steps)} failed step(s):")
        for step_num, step_name in failed_steps:
            log_info(f"  Step {step_num}: {step_name}")
        success = False
    else:
        log_success("All steps completed successfully!")
        success = True

    log_info(f"Total time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")

    return success


def parse_step_range(step_str):
    """Parse step range string like '1', '2-5', etc."""
    if '-' in step_str:
        start, end = step_str.split('-')
        return int(start), int(end)
    else:
        step = int(step_str)
        return step, step


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run global inequality analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                   # Run complete pipeline
  python run_all.py --step 1          # Run only step 1
  python run_all.py --step 2-5        # Run steps 2 through 5
  python run_all.py --skip-stata      # Skip Stata (use existing data)
  python run_all.py --list            # List all steps

Steps:
  1. Fetch WID data (Stata)
  2. Create per capita adjustments
  3. Harmonize PIP and WID data
  4. Analyze inequality decomposition
  5. Create comparison plots
  6. Compare country MLDs
  7. Run counterfactual analysis
  8. Analyze shape vs mean
        """
    )

    parser.add_argument(
        '--step',
        type=str,
        help='Step number or range to run (e.g., "1" or "2-5")'
    )

    parser.add_argument(
        '--skip-stata',
        action='store_true',
        help='Skip step 1 (Stata data fetch), use existing data'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all pipeline steps and exit'
    )

    args = parser.parse_args()

    # List steps
    if args.list:
        print("\nPipeline steps:")
        for i, (name, _) in enumerate(PIPELINE_STEPS, 1):
            print(f"  {i}. {name}")
        print()
        return

    # Determine step range
    if args.step:
        start_step, end_step = parse_step_range(args.step)

        if start_step < 1 or end_step > len(PIPELINE_STEPS):
            log_error(f"Invalid step range. Must be between 1 and {len(PIPELINE_STEPS)}")
            return
    else:
        start_step = 1
        end_step = len(PIPELINE_STEPS)

    # Run pipeline
    success = run_pipeline(
        start_step=start_step,
        end_step=end_step,
        skip_stata=args.skip_stata
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
