#!/usr/bin/env python3

import argparse
import subprocess
import pandas as pd
import tempfile
import os
import logging
import sys
import shutil
import csv # Import csv for quoting constants
import time # For timestamp

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MCQ_PARSER_SCRIPT = os.path.join(SCRIPT_DIR, "mcq_parser", "mcq_parser.py")
INPUT_FILE_DEFAULT = os.path.join(SCRIPT_DIR, "sample_manual_parsed.tsv")
LOG_FILE_DEFAULT = os.path.join(SCRIPT_DIR, "parser_comparison.log")

TARGET_COL = "model_output_text"
MANUAL_COL = "manual_label"
REFERENCE_COL = "answer"

# --- Logging Setup ---
# Remove existing handlers to avoid duplication if script is run multiple times
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()

# Setup basic config (handlers added later based on args)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def setup_file_logger(log_file_path):
    """Sets up the file logger, ensuring the directory exists."""
    log_file_path = os.path.abspath(log_file_path)
    logger = logging.getLogger() # Get root logger
    
    # Remove existing file handlers to prevent duplicates if called again
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
            
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
             os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up log file handler for {log_file_path}: {e}")
        # Log to console only if file setup fails
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
             logger.addHandler(logging.StreamHandler(sys.stdout))


def run_parser(args, input_file, temp_output_path):
    """Constructs and runs the mcq_parser.py command and determines processed file path."""
    cmd = [
        sys.executable,
        MCQ_PARSER_SCRIPT,
        "--input", input_file,
        "--output", temp_output_path,
        "--target", TARGET_COL,
        "--input-format", "tsv",
        "--delimiter", "\t",
        "--handle-x", args.handle_x
        # Add other relevant args like option_format, option_range if needed
    ]

    output_ext = os.path.splitext(temp_output_path)[1].lower()
    expected_output_delim = '\t' if output_ext == '.tsv' else ','

    is_llm_mode = False
    if args.mode == "llm":
        cmd.extend(["--use-llm", "--skip-regex", "--model", args.llm_model])
        if args.enable_cot: cmd.append("--enable-cot")
        is_llm_mode = True
    elif args.mode == "both":
        cmd.extend(["--use-llm", "--model", args.llm_model])
        if args.enable_cot: cmd.append("--enable-cot")
        is_llm_mode = True

    if args.calculate_metrics:
        cmd.extend(["--reference", REFERENCE_COL])
        is_metrics_run = True
    else:
        is_metrics_run = False

    logging.info(f"Running command: {' '.join(cmd)}")
    if is_llm_mode:
        logging.info("Starting mcq_parser.py execution (LLM mode selected, this may take time)...")
    else:
        logging.info("Starting mcq_parser.py execution...")

    try:
        # Run the command, check for errors. Don't capture/log full stdout/stderr by default.
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        # Log specific warnings or summaries from stdout if needed, but not the whole thing
        # Example: You could parse process.stdout for key lines like "Total options found..."
        if process.stderr:
             # Log stderr only if it's not empty, as it might contain useful warnings/errors
             logging.warning("mcq_parser.py stderr:\n" + process.stderr.strip())
        logging.info("mcq_parser.py execution completed successfully.")

        # --- Determine Actual Processed File Path ---
        base_output_path = os.path.splitext(temp_output_path)[0]
        processed_suffix = "_processed_data" + output_ext
        expected_processed_path = f"{base_output_path}{processed_suffix}"

        if os.path.exists(expected_processed_path):
            actual_processed_path = expected_processed_path
            logging.info(f"Found expected processed data file: {actual_processed_path}")
            return actual_processed_path, expected_output_delim
        elif not is_metrics_run and os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
             logging.warning(f"Expected processed file '{expected_processed_path}' not found. Falling back to check original output path '{temp_output_path}'.")
             actual_processed_path = temp_output_path
             return actual_processed_path, expected_output_delim
        else:
            logging.error(f"Could not find processed data file. Checked for '{expected_processed_path}' and (if no metrics) '{temp_output_path}'")
            return None, None

    except subprocess.CalledProcessError as e:
        logging.error(f"mcq_parser.py execution failed with exit code {e.returncode}.")
        logging.error("Command: " + ' '.join(e.cmd))
        # Log stdout/stderr only on error for debugging
        if e.stdout: logging.error("stdout:\n" + e.stdout.strip())
        if e.stderr: logging.error("stderr:\n" + e.stderr.strip())
        return None, None
    except FileNotFoundError:
        logging.error(f"Error: Could not find Python interpreter '{sys.executable}' or script '{MCQ_PARSER_SCRIPT}'.")
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while running mcq_parser.py: {e}")
        return None, None


def compare_outputs(original_df, processed_df, processed_col, manual_col):
    """Compares the processed column with the manual column and logs mismatches."""
    mismatch_count = 0
    logging.info(f"Comparing processed column '{processed_col}' with manual column '{manual_col}'...")
    logging.info("--- Mismatched Lines ---") # Keep this header

    if len(original_df) != len(processed_df):
        logging.warning(f"Original ({len(original_df)}) and processed ({len(processed_df)}) files have different lengths. Comparison might be inaccurate.")

    comparison_len = min(len(original_df), len(processed_df))

    for i in range(comparison_len):
        try:
            processed_val = processed_df.iloc[i][processed_col]
            manual_val = original_df.iloc[i][manual_col]
            processed_str = str(processed_val).strip() if pd.notna(processed_val) else ""
            manual_str = str(manual_val).strip() if pd.notna(manual_val) else ""

            if processed_str != manual_str:
                mismatch_count += 1
                # Log individual mismatches - this is important detail
                logging.info(f"Line {i+2}: Processed[{processed_col}]=\"{processed_str}\" != Manual[{manual_col}]=\"{manual_str}\"")
        except KeyError as e:
            logging.error(f"Column '{e}' not found during comparison at index {i}. Stopping comparison.")
            return -1 # Indicate error
        except IndexError:
             logging.error(f"Index {i} out of bounds during comparison. Stopping.")
             return -1

    # Keep the summary count
    # logging.info("--- Comparison Summary ---") # Header removed for brevity
    # logging.info(f"Total mismatched lines: {mismatch_count}")
    return mismatch_count


def main():
    parser = argparse.ArgumentParser(description="Run mcq_parser.py and compare its output against manual labels.")
    parser.add_argument("-m", "--mode", choices=['regex', 'llm', 'both'], default='regex', help="Parsing mode.")
    parser.add_argument("-l", "--llm_model", default='llama3:latest', help="Ollama model name for LLM modes.")
    parser.add_argument("-c", "--calculate-metrics", action='store_true', help=f"Calculate accuracy metrics using '{REFERENCE_COL}'.")
    parser.add_argument("--input", default=INPUT_FILE_DEFAULT, help=f"Path to the input TSV file.")
    parser.add_argument("--log", default=LOG_FILE_DEFAULT, help=f"Path to the output log file.")
    parser.add_argument("--enable-cot", action='store_true', help="Enable CoT for LLM.")
    parser.add_argument("--handle-x", choices=["none", "keep", "random"], default="keep", help="How mcq_parser.py should handle 'x'.")

    args = parser.parse_args()

    # Setup file logging based on args.log
    setup_file_logger(args.log)

    # --- Basic Arg Validation ---
    # (Keep validation logic)
    if args.mode in ['llm', 'both'] and not args.llm_model: logging.error("LLM model needed for mode '%s'", args.mode); sys.exit(1)
    if not os.path.exists(args.input): logging.error(f"Input file not found: {args.input}"); sys.exit(1)
    if not os.path.exists(MCQ_PARSER_SCRIPT): logging.error(f"mcq_parser.py script not found: {MCQ_PARSER_SCRIPT}"); sys.exit(1)

    # --- Log Run Parameters ---
    start_time = time.time()
    run_params = f"Mode: {args.mode}"
    if args.mode in ['llm', 'both']: run_params += f", Model: {args.llm_model}, CoT: {args.enable_cot}"
    run_params += f", Metrics: {args.calculate_metrics}, HandleX: {args.handle_x}"
    logging.info(f"\n\n===== Starting Comparison Run ({run_params}) =====")
    logging.info(f"Input: {args.input}, Log: {os.path.abspath(args.log)}")

    temp_output_file = None
    actual_processed_path = None
    processed_file_delimiter = None
    exit_code = 0
    mismatches = -1 # Default to error indicator

    try:
        # Use .tsv suffix to force mcq_parser to SAVE as TSV
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp_out:
            temp_output_file = tmp_out.name
        # logging.info(f"Using temporary metrics/output file: {temp_output_file}") # Less verbose

        # --- Run the Parser ---
        actual_processed_path, processed_file_delimiter = run_parser(args, args.input, temp_output_file)

        if actual_processed_path is None:
             logging.error("mcq_parser.py failed or output not found.")
             exit_code = 1
             sys.exit(exit_code)

        # --- Read Files ---
        # logging.info("Reading original input file...") # Less verbose
        original_df = pd.read_csv(args.input, sep='\t', keep_default_na=False, na_values=[''])
        # logging.info(f"Reading processed output file '{actual_processed_path}'...") # Less verbose
        processed_df = pd.read_csv(
            actual_processed_path, sep=processed_file_delimiter, engine='python',
            quoting=csv.QUOTE_MINIMAL, keep_default_na=False, na_values=['']
        )
        # logging.info(f"Successfully read files.") # Less verbose

        # --- Validate Columns ---
        required_original_cols = {MANUAL_COL}
        if args.calculate_metrics: required_original_cols.add(REFERENCE_COL)
        missing_original_cols = required_original_cols - set(original_df.columns)
        if missing_original_cols: raise ValueError(f"Missing columns in original input: {missing_original_cols}")
        required_processed_cols = {TARGET_COL}
        missing_processed_cols = required_processed_cols - set(processed_df.columns)
        if missing_processed_cols: raise ValueError(f"Missing column '{TARGET_COL}' in processed file")

        # --- Compare Outputs ---
        mismatches = compare_outputs(original_df, processed_df, TARGET_COL, MANUAL_COL)
        if mismatches < 0: exit_code = 1 # Comparison itself failed

        # --- Optional: Log Metrics ---
        if args.calculate_metrics:
            metrics_file_path = temp_output_file
            if os.path.exists(metrics_file_path) and os.path.getsize(metrics_file_path) > 0:
                try:
                    # Read metrics for summary, don't log the whole file
                    metrics_df = pd.read_csv(metrics_file_path, sep=processed_file_delimiter) # Use same delimiter
                    total_correct = metrics_df['correct'].sum()
                    total_in_calc = metrics_df['count_in_calc'].sum()
                    if total_in_calc > 0:
                        overall_acc = (total_correct / total_in_calc) * 100
                        logging.info(f"Metrics Summary: Overall Accuracy={overall_acc:.2f}% ({total_correct}/{total_in_calc})")
                    else:
                         logging.info("Metrics Summary: No valid predictions for accuracy calculation.")
                except Exception as e:
                    logging.error(f"Could not read/parse metrics file '{metrics_file_path}': {e}")
            else:
                logging.warning("Metrics file not found or empty, although metrics calculation was requested.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred in the main script: {e}")
        exit_code = 1
    finally:
        # --- Cleanup ---
        if temp_output_file and os.path.exists(temp_output_file):
            try: os.remove(temp_output_file) # logging.info(f"Removed temporary file: {temp_output_file}") # Less verbose
            except OSError as e: logging.warning(f"Could not remove temporary file {temp_output_file}: {e}")
        if actual_processed_path and actual_processed_path != temp_output_file and os.path.exists(actual_processed_path):
             try: os.remove(actual_processed_path) # logging.info(f"Removed temporary processed file: {actual_processed_path}") # Less verbose
             except OSError as e: logging.warning(f"Could not remove temporary processed file {actual_processed_path}: {e}")

        end_time = time.time()
        duration = end_time - start_time
        # --- Final Summary Log ---
        summary_status = "FAILED" if exit_code != 0 else "COMPLETED"
        summary_mismatches = f"Mismatches={mismatches}" if mismatches >= 0 else "Comparison_Error"
        logging.info(f"===== Run {summary_status} ({run_params}) Duration={duration:.2f}s {summary_mismatches} =====")

        sys.exit(exit_code)


if __name__ == "__main__":
    main()