import pandas as pd
import sys
import os
import argparse
import string
import traceback

def clear_screen():
    """Clears the terminal screen based on the operating system."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_row(display_identifier, source_text, current_parsed, parsed_col_name, total_rows, current_row_num, no_index_mode=False):
    """
    Displays the current row information in a formatted way.
    
    Args:
        display_identifier: Value from the index column or row number
        source_text: The source text to be displayed
        current_parsed: Current value in the parsed column
        parsed_col_name: Name of the column containing parsed data
        total_rows: Total number of rows in the dataset
        current_row_num: Current row being processed (0-based)
        no_index_mode: If True, uses row number as identifier
    """
    print("-" * 60)
    print(f"Processing Row {current_row_num + 1} / {total_rows}")
    if no_index_mode:
        print(f"Row Number (0-based): {display_identifier}")
    else:
        print(f"Index Column Value: {str(display_identifier)}")
    print(f"Current '{parsed_col_name}': '{current_parsed if pd.notna(current_parsed) and current_parsed else '<empty>'}'")
    print("\nSource Column Content:")
    max_width = 80
    words = str(source_text).split()
    lines = []
    current_line = ""
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + len(word) + 1 <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    print("\n".join(lines))
    print("-" * 60)

def generate_valid_options(option_format: str, option_range: str, no_answer_token: str) -> list[str]:
    """
    Generates a list of valid response options based on format and range.
    
    Args:
        option_format: Format of options ("letter" or "number")
        option_range: Range of options (e.g., "a-d", "1-6")
        no_answer_token: Token used to indicate no valid answer
        
    Returns:
        List of valid option characters
    """
    valid_set = set()
    if option_format == "letter":
        valid_set = set(string.ascii_lowercase[:6]) if option_range == "a-f" else set(string.ascii_lowercase[:4])
    elif option_format == "number":
        valid_set = set(string.digits[1:7]) if option_range == "1-6" else set(string.digits[1:5])
    else:
        print(f"Warning: Unknown option_format '{option_format}'. Defaulting to letters 'a-d'.")
        valid_set = set(string.ascii_lowercase[:4])

    if no_answer_token:
        valid_set.add(no_answer_token.lower())
    return sorted(list(valid_set))

def determine_file_params(file_path: str, specified_format: str = None, specified_delimiter: str = None) -> tuple[str, str]:
    """
    Determines file format and delimiter based on file extension and specifications.
    
    Args:
        file_path: Path to the file
        specified_format: Explicitly specified format (overrides auto-detection)
        specified_delimiter: Explicitly specified delimiter (overrides defaults)
        
    Returns:
        Tuple of (format, delimiter)
    """
    fmt = specified_format
    delim = specified_delimiter

    if fmt is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv': fmt = 'csv'
        elif ext == '.tsv': fmt = 'tsv'
        else: fmt = 'infer'

    if delim is None:
        if fmt == 'csv': delim = ','
        elif fmt == 'tsv': delim = '\t'
        else: delim = None

    if delim == 'tab': delim = '\t'
    elif delim == 'pipe': delim = '|'

    return fmt, delim


def parse_arguments():
    """
    Parses command-line arguments and performs basic validation.
    
    Returns:
        Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Manually review text from an input file and assign/update a parsed answer in an output file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, dest='input_file',
                        help="Path to the input data file (CSV, TSV, TXT) containing source text.")
    parser.add_argument("--output", required=True, dest='output_file',
                        help="Path to the output file (CSV, TSV) where manual parsed answers are stored/updated.")
    parser.add_argument("--source-col", required=True, dest='source_col_name',
                        help="Column name in the --input file containing the text to display.")
    parser.add_argument("--parsed-col", required=True, dest='parsed_col_name',
                        help="Column name in the --output file to store/update the manual answer.")

    parser.add_argument("--index-col", default=None, dest='index_col_name',
                        help="Column name for aligning rows (must be unique & in both files if --output exists). "
                             "Use 'Unnamed: 0' for pandas default index header. Required unless --no-index is used.")
    parser.add_argument("--no-index", action="store_true",
                        help="Indicate input file has NO dedicated index column. Process sequentially. Cannot reliably merge with existing output if row counts differ.")

    parser.add_argument("--input-format", choices=["csv", "tsv", "txt", "infer"], default=None,
                       help="Format of input file. If None, attempts auto-detect from extension.")
    parser.add_argument("--delimiter", default=None,
                       help="Custom delimiter for input/output files (e.g., ',', 'tab', '|'). If None, inferred from format/extension.")
    parser.add_argument("--option-format", choices=["letter", "number"], default="letter",
                       help="Expected format of options ('letter' or 'number'). Determines valid inputs.")
    parser.add_argument("--option-range", choices=["a-d", "a-f", "1-4", "1-6"], default="a-d",
                       help="Expected range of options (e.g., 'a-d', '1-6'). Determines valid inputs.")
    parser.add_argument("--no-answer-token", default="x",
                        help="Special token allowed as input to indicate no valid answer.")

    args = parser.parse_args()

    # Validation: Ensure index method is specified
    if not args.no_index and args.index_col_name is None:
        parser.error("You must specify either --index-col OR use the --no-index flag.")
    if args.no_index and args.index_col_name is not None:
         print(f"Warning: --index-col '{args.index_col_name}' is ignored because --no-index was specified.")
         args.index_col_name = None # Ensure it's None internally if no_index is true

    if args.delimiter == 'tab': args.delimiter = '\t'
    elif args.delimiter == 'pipe': args.delimiter = '|'
    return args

def main():
    """
    Main function that handles the manual parsing workflow:
    1. Load and validate input/output files
    2. Merge existing parsed answers if available
    3. Handle interactive review and update of each row
    4. Save final results
    """
    args = parse_arguments()

    # Determine valid answers based on configured options
    dynamic_valid_answers = generate_valid_options(args.option_format, args.option_range, args.no_answer_token)
    valid_answers_str = "/".join(dynamic_valid_answers)
    print(f"Expecting manual inputs from: [{valid_answers_str}]")

    # Determine file formats and delimiters
    input_format, input_delim = determine_file_params(args.input_file, args.input_format, args.delimiter)
    output_format, output_delim = determine_file_params(args.output_file, None, args.delimiter)

    # Load input data
    try:
        print(f"Reading input file: '{args.input_file}' (Format: {input_format.upper()}, Delimiter: '{repr(input_delim)}')")
        read_kwargs = {'sep': input_delim, 'dtype': str, 'keep_default_na': False}
        if input_format == 'infer' or input_delim is None:
            read_kwargs['engine'] = 'python'
            read_kwargs.pop('sep', None)

        df_input = pd.read_csv(args.input_file, **read_kwargs)
        print(f"Successfully loaded {len(df_input)} rows from '{args.input_file}'.")

        # Validate input columns
        required_input_cols = {args.source_col_name}
        if not args.no_index:
             required_input_cols.add(args.index_col_name)

        missing_input_cols = required_input_cols - set(df_input.columns)
        if missing_input_cols:
            raise ValueError(f"Missing required columns in the input file '{args.input_file}': {missing_input_cols}. Available: {list(df_input.columns)}")

        # If using index column, set index for potential merging
        if not args.no_index:
            if not df_input[args.index_col_name].is_unique:
                print(f"Warning: Index column '{args.index_col_name}' in input file '{args.input_file}' is not unique. Merging behavior might be unpredictable.")
            df_input = df_input.set_index(args.index_col_name, drop=False)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found."); sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{args.input_file}' is empty."); sys.exit(1)
    except ValueError as ve:
        print(f"Error: {ve}"); sys.exit(1)
    except Exception as e:
        print(f"Error loading/validating input file '{args.input_file}': {e}"); traceback.print_exc(); sys.exit(1)

    # Load or initialize output data and merge with input
    df_output_existing = None
    df_merged = None

    if args.no_index:
        # No Index Mode - rely on row position
        print("Running in --no-index mode. Row order is critical.")
        df_merged = df_input.copy()
        if os.path.exists(args.output_file):
            print(f"Attempting to load existing answers from '{args.output_file}' assuming row-by-row alignment...")
            try:
                read_kwargs_out = {'sep': output_delim, 'dtype': str, 'keep_default_na': False}
                if output_format == 'infer' or output_delim is None:
                    read_kwargs_out['engine'] = 'python'; read_kwargs_out.pop('sep', None)
                df_output_existing = pd.read_csv(args.output_file, **read_kwargs_out)

                if len(df_output_existing) != len(df_input):
                     print(f"Warning: Row count mismatch! Input ({len(df_input)}) vs Output ({len(df_output_existing)}). Cannot reliably load existing answers. Starting fresh.")
                     df_merged[args.parsed_col_name] = ''
                elif args.parsed_col_name not in df_output_existing.columns:
                     print(f"Parsed column '{args.parsed_col_name}' not found in existing output. Adding it.")
                     df_merged[args.parsed_col_name] = ''
                else:
                     print("Loading parsed answers positionally.")
                     df_merged[args.parsed_col_name] = df_output_existing[args.parsed_col_name].values
            except Exception as e:
                print(f"Error loading existing output file '{args.output_file}' in --no-index mode: {e}. Starting fresh.")
                df_merged[args.parsed_col_name] = ''
        else:
             print(f"Output file '{args.output_file}' not found. Adding new parsed column '{args.parsed_col_name}'.")
             df_merged[args.parsed_col_name] = ''
    else:
        # Index Column Mode - align by index value
        print(f"Aligning input data with output using index '{args.index_col_name}'...")
        if os.path.exists(args.output_file):
            try:
                read_kwargs_out = {'sep': output_delim, 'dtype': str, 'keep_default_na': False}
                if output_format == 'infer' or output_delim is None:
                    read_kwargs_out['engine'] = 'python'; read_kwargs_out.pop('sep', None)
                df_output_existing = pd.read_csv(args.output_file, **read_kwargs_out)

                if args.index_col_name not in df_output_existing.columns:
                    raise ValueError(f"Index column '{args.index_col_name}' not found in existing output file '{args.output_file}'.")
                if not df_output_existing[args.index_col_name].is_unique:
                    print(f"Warning: Index column '{args.index_col_name}' in output file '{args.output_file}' is not unique.")

                df_output_existing = df_output_existing.set_index(args.index_col_name, drop=False)

                if args.parsed_col_name not in df_output_existing.columns:
                    print(f"Parsed column '{args.parsed_col_name}' not found in existing output. Adding it.")
                    df_merged = df_input.copy()
                    df_merged[args.parsed_col_name] = ''
                else:
                    # Perform left join, keeping all rows from input
                    df_merged = df_input.join(df_output_existing[[args.parsed_col_name]], how='left')
                    print("Merge complete.")

            except FileNotFoundError:
                 df_merged = df_input.copy(); df_merged[args.parsed_col_name] = ''
            except pd.errors.EmptyDataError:
                 print("Output file empty."); df_merged = df_input.copy(); df_merged[args.parsed_col_name] = ''
            except ValueError as ve: print(f"Error: {ve}"); sys.exit(1)
            except Exception as e:
                 print(f"Error loading output '{args.output_file}': {e}. Starting fresh."); df_merged = df_input.copy(); df_merged[args.parsed_col_name] = ''
        else:
            print(f"Output file '{args.output_file}' not found. Adding new parsed column '{args.parsed_col_name}'.")
            df_merged = df_input.copy()
            df_merged[args.parsed_col_name] = ''

    # Ensure parsed_col is string and fill NaNs from join/init
    df_merged[args.parsed_col_name] = df_merged[args.parsed_col_name].fillna('').astype(str)

    # Prepare for interactive loop
    if not args.no_index:
        df_merged = df_merged.reset_index(drop=True)

    total_rows = len(df_merged)
    quit_early = False
    start_row = 0

    # Determine resume point
    try:
        unparsed_ilocs = df_merged[~df_merged[args.parsed_col_name].isin(dynamic_valid_answers)].index
        if not unparsed_ilocs.empty:
             first_unparsed_iloc = unparsed_ilocs[0]
             if args.no_index:
                 display_id_for_resume = first_unparsed_iloc; id_type = "row number"
             else:
                 display_id_for_resume = df_merged.iloc[first_unparsed_iloc][args.index_col_name]
                 id_type = f"index '{args.index_col_name}'"

             resume_choice = input(f"Found rows without valid answers ({valid_answers_str}) in '{args.parsed_col_name}', starting at {id_type} '{display_id_for_resume}' (row {first_unparsed_iloc + 1}). Resume? (y/n, default y): ").lower().strip()
             if resume_choice == '' or resume_choice == 'y':
                 start_row = first_unparsed_iloc
                 print(f"Resuming from row {start_row + 1}...")
             else:
                 print("Starting from row 1.")
        else:
            print(f"All rows appear valid ({valid_answers_str}) in '{args.parsed_col_name}'. Reviewing from start.")
    except Exception as e:
        print(f"Warning: Could not determine resume point ({e}). Starting from the beginning.")

    # Main interactive row processing loop
    for i in range(start_row, total_rows):
        row_data = df_merged.iloc[i]
        pandas_label = i

        # Determine display identifier
        if args.no_index:
            display_id = i
        else:
            display_id = row_data[args.index_col_name]

        source_text_display = row_data[args.source_col_name]
        current_parsed_val = row_data[args.parsed_col_name]

        clear_screen()
        display_row(display_id, source_text_display, current_parsed_val, args.parsed_col_name, total_rows, i, args.no_index)

        # Get user input for this row
        while True:
            prompt = f"Enter parsed answer ({valid_answers_str}), 's' to skip, 'q' to save & quit: "
            user_input = input(prompt).lower().strip()

            if user_input in dynamic_valid_answers:
                df_merged.loc[pandas_label, args.parsed_col_name] = user_input
                print(f"-> Set '{args.parsed_col_name}' to '{user_input}'")
                break
            elif user_input == 's':
                print(f"-> Skipped ('{args.parsed_col_name}' remains '{current_parsed_val}').")
                break
            elif user_input == 'q':
                print("Saving progress and quitting..."); quit_early = True; break
            else:
                print(f"Invalid input. Please enter one of [{valid_answers_str}], 's', or 'q'.")

        if quit_early: break

    # Save results to output file
    try:
        df_to_save = df_merged.copy()

        print(f"\nSaving {'progress' if quit_early else 'final results'} to '{args.output_file}' (Delimiter: '{repr(output_delim)}')...")

        df_to_save.to_csv(args.output_file, sep=output_delim, index=False, header=True, encoding='utf-8')

        print(f"Successfully saved.")

    except Exception as e:
        print(f"\nError saving results to '{args.output_file}': {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()