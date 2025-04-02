#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# set -e # Temporarily disable exit on error to allow logging failures

# --- Configuration ---
SCRIPT_NAME="mcq_parser.py"
INPUT_FILE="sample.tsv"
TARGET_COL="gpt"
REFERENCE_COL="answer"
FEW_SHOT_FILE="example_few_shot.txt"
LOG_FILE="test.log" # Define the log file name

# --- Model Selection ---
DEFAULT_MODEL="llama3:latest"
if [ -n "$1" ]; then
    TEST_MODEL="$1"
    echo "Using specified model: $TEST_MODEL"
else
    TEST_MODEL="$DEFAULT_MODEL"
    echo "No model specified, using default: $TEST_MODEL"
fi
echo # Newline

# Output files for results
OUTPUT_REGEX="results_regex.csv"
OUTPUT_LLM_FALLBACK="results_llm_fallback.csv"
OUTPUT_LLM_ONLY="results_llm_only.csv"
OUTPUT_LLM_FEWSHOT="results_llm_fewshot.csv"
OUTPUT_HANDLE_X_KEEP="results_handle_x_keep.csv"

# Intermediate processed data files
PROC_REGEX="${OUTPUT_REGEX%.csv}_processed_data.csv"
PROC_LLM_FALLBACK="${OUTPUT_LLM_FALLBACK%.csv}_processed_data.csv"
PROC_LLM_ONLY="${OUTPUT_LLM_ONLY%.csv}_processed_data.csv"
PROC_LLM_FEWSHOT="${OUTPUT_LLM_FEWSHOT%.csv}_processed_data.csv"
PROC_HANDLE_X_KEEP="${OUTPUT_HANDLE_X_KEEP%.csv}_processed_data.csv"

# --- Initialize Log File ---
# Clear the log file at the start of the run
echo "Initializing log file: $LOG_FILE"
echo "" > "$LOG_FILE"
echo "Test Run Started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "============================================" >> "$LOG_FILE"

# --- Helper Function ---
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file '$1' not found!" | tee -a "$LOG_FILE" # Log error
        exit 1
    fi
}

run_parser() {
    local test_desc="$1"
    local python_args="${@:2}" # Get arguments for the python script
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_header_prefix="[$timestamp] --- Starting Test: $test_desc"
    local log_header_suffix=" ---"
    local log_header="$log_header_prefix$log_header_suffix"
    local is_llm_run=0

    # Check if --use-llm is present in the arguments
    if [[ "$python_args" == *"--use-llm"* ]]; then
      is_llm_run=1
      # Add model name to header if it's an LLM run
      log_header="$log_header_prefix [Model: $TEST_MODEL]$log_header_suffix"
    fi

    # --- Log Header ---
    echo "" | tee -a "$LOG_FILE" # Add empty line for spacing in log and console
    echo "$log_header" | tee -a "$LOG_FILE"
    echo "[$timestamp] Command: python $SCRIPT_NAME $python_args" | tee -a "$LOG_FILE"
    echo "[$timestamp] -----------------------------------------------------" | tee -a "$LOG_FILE"

    # --- Execute and Log ---
    # Execute the python script, append both stdout and stderr to the log file,
    # and also display stdout/stderr on the console using tee.
    if python "$SCRIPT_NAME" $python_args >> "$LOG_FILE" 2>&1 ; then
        # Success case (Python script exited with 0)
        echo "[$timestamp] --- Test '$test_desc' Completed Successfully ---" | tee -a "$LOG_FILE"
        return 0 # Indicate success
    else
        # Failure case (Python script exited with non-zero)
        local exit_code=$?
        echo "[$timestamp] !!! Test '$test_desc' FAILED with exit code $exit_code !!!" | tee -a "$LOG_FILE"
        return $exit_code # Propagate the error code
    fi
    # Note: Removed the tee pipe from the main command to capture exit code correctly.
    # Logging inside the if/else ensures output goes to the file. Stdout/stderr
    # from the python script are redirected directly to the log file.
    # If you *also* want to see the python script's live output on the console,
    # you'd need a more complex setup, perhaps involving process substitution or named pipes.
    # Keeping it simple: All python output goes directly to the log file in this version.
    # --- Revision: Let's use tee again but capture exit code properly ---
    # Command > >(tee -a log) 2> >(tee -a log >&2) - This is complex.
    # Let's stick to the simpler approach first: Python output ONLY to log file when run via script.

    # --- REVISED Execute and Log ---
    # Redirect stdout and stderr to the log file. Check exit status after.
    python "$SCRIPT_NAME" $python_args >> "$LOG_FILE" 2>&1
    local exit_code=$? # Capture exit code IMMEDIATELY

    if [ $exit_code -eq 0 ]; then
        echo "[$timestamp] --- Test '$test_desc' Completed Successfully ---" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[$timestamp] !!! Test '$test_desc' FAILED with exit code $exit_code (See $LOG_FILE for details) !!!" | tee -a "$LOG_FILE"
        # Optionally, print last few lines of log on failure
        echo "[$timestamp] Last few lines of log:" | tee -a "$LOG_FILE"
        tail -n 10 "$LOG_FILE" | sed "s/^/[$timestamp] LOG: /" | tee -a "$LOG_FILE" # Indent log lines
        return $exit_code
    fi

    echo # Newline for readability (only to console)
    sleep 1
}


# --- Pre-checks ---
echo "Performing pre-checks..." | tee -a "$LOG_FILE"
check_file_exists "$SCRIPT_NAME"
check_file_exists "$INPUT_FILE"
check_file_exists "$FEW_SHOT_FILE"
echo "Pre-checks passed." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"


# --- Cleanup Function ---
cleanup() {
  echo "Cleaning up generated CSV/TSV files..." | tee -a "$LOG_FILE"
  # DO NOT delete the log file here
  rm -f "$OUTPUT_REGEX" "$PROC_REGEX" \
        "$OUTPUT_LLM_FALLBACK" "$PROC_LLM_FALLBACK" \
        "$OUTPUT_LLM_ONLY" "$PROC_LLM_ONLY" \
        "$OUTPUT_LLM_FEWSHOT" "$PROC_LLM_FEWSHOT" \
        "$OUTPUT_HANDLE_X_KEEP" "$PROC_HANDLE_X_KEEP"
  echo "Cleanup complete." | tee -a "$LOG_FILE"
}

# Register cleanup to run on exit
trap cleanup EXIT

# Variable to track overall test status
overall_status=0

# --- Run Tests ---

# Test 1: Basic Regex Extraction + Accuracy (Default settings)
run_parser "Regex Only (Defaults)" \
    "--input $INPUT_FILE --output $OUTPUT_REGEX --target $TARGET_COL --reference $REFERENCE_COL" || overall_status=$?
if [ $overall_status -eq 0 ]; then
    check_file_exists "$OUTPUT_REGEX"
    check_file_exists "$PROC_REGEX"
    echo ">>> Regex Only Test: Output files created." | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Conditionally run LLM tests
if [ $overall_status -eq 0 ]; then
    echo "INFO: The following LLM tests require Ollama to be running with model '$TEST_MODEL'." | tee -a "$LOG_FILE"
    echo "INFO: Ensure '$TEST_MODEL' is pulled (e.g., 'ollama pull $TEST_MODEL')." | tee -a "$LOG_FILE"
    echo "INFO: They may take several minutes to complete." # Console only
    read -p "Proceed with LLM tests? (y/N): " proceed_llm
    if [[ "$proceed_llm" != "y" && "$proceed_llm" != "Y" ]]; then
        echo "Skipping LLM tests." | tee -a "$LOG_FILE"
        llm_tests_skipped=true
    else
        llm_tests_skipped=false
        # Test 2: Regex + LLM Fallback
        run_parser "Regex + LLM Fallback" \
            "--input $INPUT_FILE --output $OUTPUT_LLM_FALLBACK --target $TARGET_COL --reference $REFERENCE_COL --use-llm --model $TEST_MODEL" || overall_status=$?
        if [ $overall_status -eq 0 ]; then
            check_file_exists "$OUTPUT_LLM_FALLBACK"
            check_file_exists "$PROC_LLM_FALLBACK"
            echo ">>> Regex + LLM Fallback Test: Output files created." | tee -a "$LOG_FILE"
        fi
        echo "" | tee -a "$LOG_FILE"

        # Test 3: LLM Only (Skip Regex)
        if [ $overall_status -eq 0 ]; then
            run_parser "LLM Only (Skip Regex)" \
                "--input $INPUT_FILE --output $OUTPUT_LLM_ONLY --target $TARGET_COL --reference $REFERENCE_COL --use-llm --skip-regex --model $TEST_MODEL" || overall_status=$?
            if [ $overall_status -eq 0 ]; then
                check_file_exists "$OUTPUT_LLM_ONLY"
                check_file_exists "$PROC_LLM_ONLY"
                echo ">>> LLM Only Test: Output files created." | tee -a "$LOG_FILE"
            fi
            echo "" | tee -a "$LOG_FILE"
        fi

        # Test 4: LLM Only with Few-Shot Examples
        if [ $overall_status -eq 0 ]; then
            run_parser "LLM Only + Few-Shot" \
                "--input $INPUT_FILE --output $OUTPUT_LLM_FEWSHOT --target $TARGET_COL --reference $REFERENCE_COL --use-llm --skip-regex --model $TEST_MODEL --examples-file $FEW_SHOT_FILE" || overall_status=$?
            if [ $overall_status -eq 0 ]; then
                check_file_exists "$OUTPUT_LLM_FEWSHOT"
                check_file_exists "$PROC_LLM_FEWSHOT"
                echo ">>> LLM Only + Few-Shot Test: Output files created." | tee -a "$LOG_FILE"
            fi
            echo "" | tee -a "$LOG_FILE"
        fi
    fi
fi

# Test 5: Different Handle-X mode (Regex only for speed)
if [ $overall_status -eq 0 ]; then
    run_parser "Regex Only (handle-x=keep)" \
        "--input $INPUT_FILE --output $OUTPUT_HANDLE_X_KEEP --target $TARGET_COL --reference $REFERENCE_COL --handle-x keep" || overall_status=$?
    if [ $overall_status -eq 0 ]; then
        check_file_exists "$OUTPUT_HANDLE_X_KEEP"
        check_file_exists "$PROC_HANDLE_X_KEEP"
        echo ">>> Handle-X Keep Test: Output files created." | tee -a "$LOG_FILE"
    fi
    echo "" | tee -a "$LOG_FILE"
fi


# --- End of Tests ---
echo "" | tee -a "$LOG_FILE"
if [ $overall_status -eq 0 ]; then
  if [ "$llm_tests_skipped" = true ]; then
    echo "Basic tests completed successfully (LLM tests skipped)." | tee -a "$LOG_FILE"
  else
    echo "All specified tests executed successfully!" | tee -a "$LOG_FILE"
  fi
else
  echo "!!! One or more tests FAILED. Check '$LOG_FILE' for details. !!!" | tee -a "$LOG_FILE"
fi
echo "============================================" >> "$LOG_FILE"
echo "Test Run Finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"

# Exit with the overall status
exit $overall_status