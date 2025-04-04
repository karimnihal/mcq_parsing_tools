import re
import string
import random
import pandas as pd
import requests
import json
import time
import os
import argparse # Ensure argparse is imported
import traceback # For better error reporting
from sklearn.metrics import accuracy_score
from typing import Optional, Dict, Any, List, Tuple, Union

def extract_predicted_option(text: str, option_format="letter", option_range="a-d"):
    """
    Attempt to extract a predicted option using regex strategies.

    Parameters:
    - text (str): The text to extract an option from
    - option_format (str): The format of options, either "letter" for a-f or "number" for 1-6
    - option_range (str): The range of options, e.g. "a-d", "a-f", "1-4", "1-6"

    Returns:
    - str or None: Extracted option or None if extraction failed
    """
    if text is None:
        return None

    text_lower = str(text).lower()

    # Determine the option characters based on format and range
    if option_format == "letter":
        if option_range == "a-f":
            valid_options = "abcdef"
        else:  # default a-d
            valid_options = "abcd"
    else:  # number format
        if option_range == "1-6":
            valid_options = "123456"
        else:  # default 1-4
            valid_options = "1234"

    # Escape special characters for regex
    escaped_options = re.escape(valid_options)

    # 1st extraction: option followed by a colon or period (e.g., "c:", "c.", "3:", "3.") - prioritize this
    pattern1 = f'(?<!\\w)([{escaped_options}])[:.]'
    match1 = re.search(pattern1, text_lower)
    if match1:
        return match1.group(1)

    # Look for phrases that strongly indicate an answer
    # Added more variations and made patterns more specific
    answer_indicators = [
        f'answer is[:\\s]+([{escaped_options}])\\b',
        f'answer[:\\s]+([{escaped_options}])\\b',
        f'\\bchoice[:\\s]+([{escaped_options}])\\b',
        f'\\boption[:\\s]+([{escaped_options}])\\b',
        f'\\bselect[:\\s]+([{escaped_options}])\\b',
        f'\\bselected[:\\s]+([{escaped_options}])\\b',
        f'\\bchose[:\\s]+([{escaped_options}])\\b',
        f'\\bchoosing[:\\s]+([{escaped_options}])\\b',
        f'\\bcorrect (?:answer is|option is)?[:\\s]+([{escaped_options}])\\b',
        f'\\b([{escaped_options}]) is correct\\b',
        f'\\bfinal answer[:\\s]+([{escaped_options}])\\b',
        f'\\bthe correct option is ([{escaped_options}])\\b'
        # removed 'the (option)' as it's too general
    ]

    for pattern in answer_indicators:
        match = re.search(pattern, text_lower)
        if match:
            # Group index might vary based on pattern structure
            if len(match.groups()) == 1:
                return match.group(1)
            elif len(match.groups()) > 1:
                 # Handle patterns with multiple capture groups if necessary,
                 # usually the last group is the target
                 return match.group(len(match.groups()))


    # 2nd extraction: standalone option character not preceded/followed by word chars
    # Make this less prioritized than strong indicators
    pattern2 = f'(?<!\\w)([{escaped_options}])(?!\\w)'
    match2 = re.search(pattern2, text_lower)
    if match2:
        return match2.group(1)

    # Special case for when text is ONLY a single option character
    if text_lower.strip() in valid_options and len(text_lower.strip()) == 1:
        return text_lower.strip()


    # All extraction methods failed
    return None


def extract_predicted_option_llm(text: str, model: str = "llama3:latest", no_answer_token: str = "x",
                             option_format="letter", option_range="a-d", max_tokens: int = 4096,
                             enable_cot: bool = False,
                             examples: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """
    Use a local LLM via Ollama to extract a single token answer, potentially using few-shot examples.
    (Implementation assumed correct from previous steps)
    """
    if text is None:
        return None

    examples = examples or [] # Ensure examples is a list

    # Determine valid options based on format and range
    if option_format == "letter":
        if option_range == "a-f":
            valid_options_list = list("abcdef")
            option_desc = "a, b, c, d, e, f"
        else:  # default a-d
            valid_options_list = list("abcd")
            option_desc = "a, b, c, d"
    else:  # number format
        if option_range == "1-6":
            valid_options_list = list("123456")
            option_desc = "1, 2, 3, 4, 5, 6"
        else:  # default 1-4
            valid_options_list = list("1234")
            option_desc = "1, 2, 3, 4"

    valid_options_set = "".join(valid_options_list) + no_answer_token
    full_option_desc = f"{option_desc}, or {no_answer_token}"

    # --- Token Estimation and Truncation ---
    prompt_base_overhead = 200
    example_token_estimate = 0
    formatted_examples = ""
    if examples:
        formatted_examples += "Here are some examples:\n--- EXAMPLES START ---\n"
        for ex in examples:
            example_line = f"Input: {ex['text']}\nOutput: {ex['option']}\n\n"
            example_token_estimate += len(example_line) // 4
            formatted_examples += example_line
        formatted_examples += "--- EXAMPLES END ---\n\n"
        example_token_estimate += len("Here are some examples:\n--- EXAMPLES START ---\n\n--- EXAMPLES END ---\n\n") // 4

    available_tokens_for_text = max_tokens - prompt_base_overhead - example_token_estimate - 50
    if available_tokens_for_text <= 0:
         print(f"Warning: Not enough tokens ({max_tokens}) for prompt base and examples. LLM call might fail or be inaccurate.")

    original_text_length_chars = len(text)
    max_text_chars = available_tokens_for_text * 4
    if original_text_length_chars > max_text_chars:
        print(f"Text too long ({original_text_length_chars // 4} est. tokens), truncating to ~{max_text_chars // 4} tokens to fit examples and prompt within {max_tokens} limit.")
        text = text[:max_text_chars] + "... [truncated]"

    # --- Create the prompt ---
    if enable_cot:
        prompt = (
            f"Extract a single answer option ({full_option_desc}) from the following text.\n"
            f"{formatted_examples}"
            f"Text:\n'''{text}'''\n\n"
            f"<think>\n"
            f"Let me examine the text carefully to find the correct answer option ({option_desc}).\n"
            f"I should follow the format shown in the examples if provided.\n"
            f"I need to look for explicit indicators like 'The answer is (option)' or similar patterns.\n"
            f"If there's no clear option or the text indicates there is no answer, I must respond with exactly '{no_answer_token}'.\n"
            f"My final output must be only the single character representing the option.\n"
            f"</think>\n\n"
            f"The answer option is:"
        )
    else:
        prompt = (
            f"Extract a single answer option ({full_option_desc}) from the following text.\n"
            f"{formatted_examples}"
            f"Text:\n'''{text}'''\n\n"
            f"If the text indicates there is no answer, or you cannot find a clear choice, respond with '{no_answer_token}'.\n"
            f"Ensure the answer is ONLY a single token among {full_option_desc}. Output only the option character."
        )

    response = call_ollama_api(prompt, model=model, valid_options=valid_options_set, max_tokens=max_tokens, enable_cot=enable_cot)
    response = response.strip().lower()

    # Validate response more strictly
    if response in valid_options_set and len(response) == 1:
        return response
    else:
        # Try the robust regex function on the LLM's response as a fallback
        print(f"LLM response '{response}' not a single valid option. Applying regex fallback...")
        fallback_option = extract_predicted_option(response, option_format, option_range)
        if fallback_option:
            print(f"Regex fallback extracted option: '{fallback_option}'")
            return fallback_option
        print(f"LLM extraction did not yield a valid single option character from '{valid_options_set}': '{response}'")
        return None


def call_ollama_api(prompt: str, model: str = "llama3:latest", valid_options="abcdx",
                max_tokens: int = 4096, enable_cot: bool = False) -> str:
    """
    Call a local LLM via Ollama API.
    (Implementation assumed correct from previous steps)
    """
    api_url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 150 if enable_cot else 10,
            "num_ctx": max_tokens
        }
    }
    if not enable_cot:
        payload["options"]["stop"] = ["\n", ".", ",", ";", ":", " ", "?"]

    max_retries = 3
    retry_delay = 3
    for attempt in range(max_retries):
        try:
            print(f"Calling Ollama API (Attempt {attempt+1}/{max_retries})...")
            response = requests.post(api_url, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("response", "")
            print(f"Ollama raw response: '{generated_text}'")
            if isinstance(generated_text, str):
                clean_text = generated_text.strip()
            else:
                print(f"Warning: Ollama response was not a string ('{generated_text}'). Treating as empty.")
                clean_text = ""
            if enable_cot and "</think>" in clean_text:
                    parts = clean_text.split("</think>")
                    final_part = parts[-1].strip()
                    final_part = re.sub(r'^(the answer option is|answer:|option:)\s*', '', final_part, flags=re.IGNORECASE).strip()
                    print(f"Extracted CoT final part: '{final_part}'")
                    return final_part
            else:
                    return clean_text
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}: Error calling Ollama API: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Ollama API call failed.")
                return ""
    print("All API attempts failed.")
    return ""


def check_ollama_availability(model: str = "llama3:latest") -> bool:
    """
    Check if Ollama is available and the specified model is installed.
    (Implementation assumed correct from previous steps)
    """
    api_url = "http://localhost:11434/api/tags"
    print(f"Checking Ollama availability and model '{model}'...")
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        models_data = response.json()
        if not isinstance(models_data, dict) or "models" not in models_data:
             print(f"Ollama API returned unexpected data format: {models_data}")
             return False
        models = models_data.get("models", [])
        if not isinstance(models, list):
            print(f"Ollama API returned unexpected 'models' format: {models}")
            return False
        available_models = [m.get("name") for m in models if isinstance(m, dict) and "name" in m]
        if model in available_models:
            print(f"Model '{model}' found.")
            return True
        else:
            print(f"Model '{model}' not found in Ollama. Available models: {available_models}")
            base_model = model.split(':')[0]
            if any(m.startswith(base_model + ':') for m in available_models):
                 print(f"Note: Found related models for '{base_model}'. Ensure you specified the correct tag.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Ollama API check failed: {e}")
        return False
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from Ollama API.")
        return False

def verbalizer(text: str, option_format="letter", option_range="a-d", no_answer_token="x"):
    """
    Normalize the text.
    (Implementation assumed correct from previous steps)
    """
    if text is None: return None
    if option_format == "letter": valid_set = set(list("abcdef" if option_range == "a-f" else "abcd"))
    else: valid_set = set(list("123456" if option_range == "1-6" else "1234"))
    full_valid_set = valid_set.union({no_answer_token})
    text_str = str(text).lower().strip()
    if text_str == no_answer_token: return no_answer_token
    if text_str in full_valid_set and len(text_str) == 1: return text_str
    text_clean = text_str.translate(str.maketrans('', '', string.punctuation))
    words = text_clean.split()
    for word in words:
        if word in full_valid_set: return word
    return None

def assign_label(row: pd.Series, target_col: str, reference_col: Optional[str], handle_x: str = "random",
              option_format: str = "letter", option_range: str = "a-d", no_answer_token: str = "x") -> Optional[str]:
    """
    Process the extracted prediction, handling 'x' and None/invalid/NA values.
    (Implementation assumed correct from previous steps)
    """
    prediction: Union[str, None] = row[target_col]
    if pd.isna(prediction): prediction = None
    if option_format == "letter": std_labels_list = list("abcdef" if option_range == "a-f" else "abcd")
    else: std_labels_list = list("123456" if option_range == "1-6" else "1234")
    std_labels_set = set(std_labels_list)

    if prediction == no_answer_token:
        if handle_x == 'none': return None
        elif handle_x == 'keep': return no_answer_token
        elif handle_x == 'random':
            if reference_col and reference_col in row and pd.notna(row[reference_col]):
                ref_label = str(row[reference_col]).strip().lower()
                possible = list(std_labels_set - {ref_label}) if ref_label in std_labels_set else std_labels_list
            else: possible = std_labels_list
            return random.choice(possible) if possible else None
        else: return None
    elif prediction in std_labels_set: return prediction
    else: # None or invalid
        if reference_col and reference_col in row and pd.notna(row[reference_col]):
             ref_label = str(row[reference_col]).strip().lower()
             possible = list(std_labels_set - {ref_label}) if ref_label in std_labels_set else std_labels_list
        else: possible = std_labels_list
        return random.choice(possible) if possible else None

def calculate_metric(df, target_col, reference_col):
    """
    Calculate accuracy metrics grouped by language and prompt.
    (Implementation assumed correct from previous steps)
    """
    results = []
    df[reference_col] = df[reference_col].astype(str).str.lower()
    valid_mask = df[target_col].notna() & (df[target_col] != 'x') # Assuming 'x' if handle_x='keep'
    filtered_df = df[valid_mask].copy()
    if filtered_df.empty:
        print("Warning: No valid predictions available for comparison after filtering.")
        return pd.DataFrame(columns=['language', 'prompt_no', 'accuracy', 'correct', 'count_in_calc', 'total_original'])
    filtered_df[target_col] = filtered_df[target_col].astype(str)
    filtered_df[reference_col] = filtered_df[reference_col].astype(str)
    print(f"Calculating accuracy based on {len(filtered_df)} valid predictions...")
    for (language, prompt), group_df in filtered_df.groupby(['language', 'prompt_no']):
        true_labels = group_df[reference_col].tolist()
        predicted_labels = group_df[target_col].tolist()
        if not true_labels: continue
        accuracy = accuracy_score(true_labels, predicted_labels)
        correct = int(accuracy * len(true_labels))
        count_valid = len(true_labels)
        original_total = len(df[(df['language'] == language) & (df['prompt_no'] == prompt)])
        results.append({'language': language, 'prompt_no': prompt, 'accuracy': round(accuracy * 100, 2),
                        'correct': correct, 'count_in_calc': count_valid, 'total_original': original_total})
    return pd.DataFrame(results)

def load_examples(examples_path: Optional[str]) -> Optional[List[Dict[str, str]]]:
    """Loads few-shot examples from a file."""
    if not examples_path or not os.path.exists(examples_path):
        if examples_path: print(f"Warning: Examples file not found at '{examples_path}'.")
        return None
    examples = []
    try:
        with open(examples_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'): continue
                if ':::' not in line:
                    print(f"Warning: Skipping invalid line {i+1} in examples file (missing ':::'): {line}")
                    continue
                parts = line.split(':::', 1)
                if len(parts) != 2:
                    print(f"Warning: Skipping invalid line {i+1} in examples file (format error): {line}")
                    continue
                text, option = parts[0].strip(), parts[1].strip().lower()
                if not text or not option:
                    print(f"Warning: Skipping invalid line {i+1} in examples file (empty text or option after split): {line}")
                    continue
                examples.append({'text': text, 'option': option})
        print(f"Loaded {len(examples)} few-shot examples from '{examples_path}'.")
        return examples
    except Exception as e:
        print(f"Error loading examples file '{examples_path}': {e}")
        return None

# ===============================================
# CORRECTED process_predictions Function
# ===============================================

def process_predictions(df, target_col, reference_col, use_llm=False, llm_model="llama3:latest",
                        handle_x="random", no_answer_token="x", option_format="letter", option_range="a-d",
                        max_tokens=4096, enable_cot=False, examples=None, skip_regex: bool = False):
    """
    Process predictions using extraction methods, verbalizer, and label assignment.

    Parameters:
    (Same as before)
    - skip_regex (bool): Whether to skip the initial regex extraction step.

    Returns:
    - DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    raw_prediction_col = target_col + '_raw'
    processed_df[raw_prediction_col] = processed_df[target_col] # Keep original prediction

    # Initialize extracted_option column, default to NaN/None
    # Use object dtype to allow storing strings and pd.NA
    processed_df['extracted_option'] = pd.Series(dtype='object')
    processed_df['extracted_option'] = pd.NA # Initialize all to NA

    # --- Step 1: Apply regex extraction (ONLY IF skip_regex is FALSE) ---
    if not skip_regex:
        print("Applying regex extraction...")
        # Apply regex and directly update the column for non-NA results
        extracted_options_series = processed_df[raw_prediction_col].apply(
            lambda text: extract_predicted_option(text, option_format, option_range)
        )
        # Update the column only where regex found something
        # Use .loc to ensure alignment and avoid potential SettingWithCopyWarning
        processed_df.loc[extracted_options_series.notna(), 'extracted_option'] = extracted_options_series.dropna()
        print(f"Regex extraction complete. Found options for {processed_df['extracted_option'].notna().sum()}/{len(processed_df)} entries.")
    else:
        print("Skipping regex extraction as requested.")
        # 'extracted_option' remains NA for all rows

    # --- Step 2: Apply LLM extraction if enabled ---
    llm_applied_count = 0
    if use_llm:
        ollama_available = check_ollama_availability(model=llm_model)
        if ollama_available:
            # Determine which rows need LLM processing
            if skip_regex:
                # If regex was skipped, LLM needs to run on ALL rows
                mask_needs_llm = pd.Series(True, index=processed_df.index)
                print(f"Using LLM '{llm_model}' for all {len(processed_df)} entries (regex skipped)...")
            else:
                # If regex ran, only run LLM where regex failed (extracted_option is still NA)
                mask_needs_llm = processed_df['extracted_option'].isna()
                num_to_process_with_llm = mask_needs_llm.sum()
                if num_to_process_with_llm > 0:
                    print(f"Using LLM '{llm_model}' for {num_to_process_with_llm} entries where regex extraction failed...")
                # If num_to_process is 0, mask_needs_llm.any() will be False below

            # Proceed only if there are rows identified for LLM processing
            if mask_needs_llm.any():
                if examples:
                    print(f"Using {len(examples)} few-shot examples.")

                # Define the function to apply LLM extraction row-wise
                def apply_llm_extraction_for_row(row_data):
                    # Added internal print for debugging which row text is processed by LLM
                    print(f"  LLM Processing text starting with: '{row_data[raw_prediction_col][:50]}...'")
                    return extract_predicted_option_llm(
                        row_data[raw_prediction_col], # Use original raw text
                        model=llm_model,
                        no_answer_token=no_answer_token,
                        option_format=option_format,
                        option_range=option_range,
                        max_tokens=max_tokens,
                        enable_cot=enable_cot,
                        examples=examples
                    )

                # Apply LLM extraction *only* to the selected rows identified by the mask
                rows_to_process = processed_df.loc[mask_needs_llm]
                print(f"Applying LLM extraction to {len(rows_to_process)} rows...")
                # Ensure apply result aligns with the original index for proper assignment
                llm_results = rows_to_process.apply(apply_llm_extraction_for_row, axis=1)
                llm_applied_count = len(llm_results) # Count how many were attempted

                # Update the 'extracted_option' column *only* for the rows where LLM ran
                processed_df.loc[mask_needs_llm, 'extracted_option'] = llm_results.fillna(pd.NA)

                print(f"LLM extraction applied to {llm_applied_count} entries. "
                      f"Total options found (Regex+LLM): {processed_df['extracted_option'].notna().sum()}/{len(processed_df)}")
            else:
                # This else block now correctly means "no rows needed LLM processing"
                if not skip_regex:
                     print("Regex extraction succeeded for all entries. Skipping LLM.")
                # No specific message needed if skip_regex was True but LLM wasn't needed (e.g., empty input df)

        else: # Ollama not available
            print(f"Warning: Ollama API not available or model '{llm_model}' not installed. Cannot use LLM.")
            if skip_regex:
                print("Warning: Regex was skipped and LLM is unavailable. No extraction will be performed.")
                # All 'extracted_option' will remain NA

    # --- Step 3: Apply verbalizer ---
    # Operates on the 'extracted_option' column
    print("Applying verbalizer for normalization...")
    processed_df['verbalized_option'] = processed_df['extracted_option'].apply(
        lambda text: verbalizer(text, option_format, option_range, no_answer_token)
    ).fillna(pd.NA) # Ensure NAs propagate
    print(f"Verbalizer complete. Valid options/tokens found for {processed_df['verbalized_option'].notna().sum()}/{len(processed_df)} entries.")

    # --- Step 4: Assign final label ---
    # Operates on the 'verbalized_option' column.
    print(f"Assigning final labels (handle_x='{handle_x}')...")
    final_label_col = target_col # Overwrite the original target column with the final label
    processed_df[final_label_col] = processed_df.apply(
        lambda row: assign_label(row, 'verbalized_option', reference_col, handle_x, option_format, option_range, no_answer_token),
        axis=1
    )
    print("Final label assignment complete.")

    return processed_df

# ===============================================
# Main Execution Block (__main__) - Ensure skip_regex is passed
# ===============================================

def main(input_path: str, output_path: str, target_col: str, reference_col: Optional[str],
         use_llm: bool = False, llm_model: str = "llama3:latest", handle_x: str = "random",
         no_answer_token: str = "x", option_format: str = "letter", option_range: str = "a-d",
         input_format: Optional[str] = None, delimiter: Optional[str] = None, max_tokens: int = 4096,
         enable_cot: bool = False, examples_path: Optional[str] = None,
         skip_regex: bool = False): # skip_regex parameter added here
    """
    Main function to process the input file, extract predictions, compute accuracy, and save results.
    """
    print("--- Starting MCQ Parser ---")
    # Added internal print to confirm flags received by main
    print(f"  main received: skip_regex={skip_regex}, use_llm={use_llm}, enable_cot={enable_cot}, handle_x={handle_x}")
    try:
        # --- File Reading ---
        start_time = time.time()
        # (File reading logic assumed correct)
        if input_format is None:
            ext = os.path.splitext(input_path)[1].lower()
            if ext == '.csv': input_format, auto_delimiter = 'csv', ','
            elif ext == '.tsv': input_format, auto_delimiter = 'tsv', '\t'
            elif ext == '.txt': input_format, auto_delimiter = 'txt', None
            else: print(f"Warning: Unknown file extension '{ext}'. Attempting to read as CSV/TSV."); input_format, auto_delimiter = 'infer', None
        else:
            input_format = input_format.lower()
            if input_format == 'csv': auto_delimiter = ','
            elif input_format == 'tsv': auto_delimiter = '\t'
            else: auto_delimiter = None
        delim_to_use = delimiter if delimiter is not None else auto_delimiter
        print(f"Reading input file: '{input_path}' (Format: {input_format.upper()}, Delimiter: '{repr(delim_to_use)}')")
        try:
            if input_format == 'infer' or delim_to_use is None:
                 print("Attempting to infer delimiter...")
                 df = pd.read_csv(input_path, sep=None, engine='python', keep_default_na=False, na_values=[''])
            else:
                 df = pd.read_csv(input_path, sep=delim_to_use, keep_default_na=False, na_values=[''])
            print(f"Successfully read {len(df)} rows.")
        except Exception as e: print(f"Error reading input file: {e}\nPlease check file path, format, encoding, and delimiter."); return None

        # --- Column Validation ---
        required_cols = {target_col, 'language', 'prompt_no'}
        if reference_col: required_cols.add(reference_col)
        missing_cols = required_cols - set(df.columns)
        if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")

        # --- Load Examples ---
        examples = load_examples(examples_path)

        # --- Prepare Columns ---
        df[target_col] = df[target_col].fillna('').astype(str)
        if reference_col: df[reference_col] = df[reference_col].fillna('').astype(str).str.lower()
        df['language'] = df['language'].astype(str)
        df['prompt_no'] = df['prompt_no'].astype(str)

        # --- Process Predictions ---
        print("\n--- Processing Predictions ---")
        print(f"Target Column: '{target_col}'")
        if reference_col: print(f"Reference Column: '{reference_col}'")
        print(f"Option Format: {option_format}, Option Range: {option_range}")
        print(f"Skip Regex: {skip_regex}") # Log skip_regex setting
        print(f"Use LLM: {use_llm}")
        if use_llm:
            print(f"LLM Model: {llm_model}, Max Tokens: {max_tokens}, CoT: {enable_cot}")
            if examples: print(f"Few-shot Examples: {len(examples)} loaded")
        print(f"Handle '{no_answer_token}' Token: {handle_x}")

        # Call process_predictions, passing the skip_regex value from main's parameter
        processed_df = process_predictions(
            df=df,
            target_col=target_col,
            reference_col=reference_col,
            use_llm=use_llm,
            llm_model=llm_model,
            handle_x=handle_x,
            no_answer_token=no_answer_token,
            option_format=option_format,
            option_range=option_range,
            max_tokens=max_tokens,
            enable_cot=enable_cot,
            examples=examples,
            skip_regex=skip_regex # Ensure this is passed
        )

        # --- Determine Output Format & Save ---
        output_ext = os.path.splitext(output_path)[1].lower()
        output_format = 'tsv' if output_ext == '.tsv' else 'csv'
        output_delim = '\t' if output_format == 'tsv' else ','

        processed_output_path = output_path.replace(output_ext, f"_processed_data{output_ext}")
        print(f"\nSaving detailed processed data to: '{processed_output_path}'")
        # Save using the determined delimiter
        processed_df.to_csv(processed_output_path, sep=output_delim, index=False, encoding='utf-8')

        # --- Calculate Metrics (if reference column provided) ---
        metric_df = None
        if reference_col:
            print("\n--- Calculating Accuracy Metrics ---")
            metric_df = calculate_metric(processed_df, target_col, reference_col)
            if metric_df is not None and not metric_df.empty:
                print(f"Saving accuracy metrics to: '{output_path}'")
                 # Save using the determined delimiter
                metric_df.to_csv(output_path, sep=output_delim, index=False, encoding='utf-8')
                # --- Print Summary ---
                print("\n--- Accuracy Summary ---")
                # (Summary printing logic)
                total_correct = metric_df['correct'].sum()
                total_in_calc = metric_df['count_in_calc'].sum()
                total_original_overall = metric_df['total_original'].sum()
                for _, row in metric_df.iterrows(): print(f"Language: {row['language']}, Prompt: {row['prompt_no']}, Accuracy: {row['accuracy']}% ({row['correct']}/{row['count_in_calc']} valid predictions)")
                if total_in_calc > 0:
                     overall_accuracy = (total_correct / total_in_calc) * 100
                     print(f"\nOverall Accuracy (based on {total_in_calc} valid predictions): {overall_accuracy:.2f}%")
                     print(f"Total entries processed: {total_original_overall}")
                     if total_original_overall > total_in_calc: print(f"({total_original_overall - total_in_calc} entries excluded due to missing/invalid prediction or handle_x settings)")
                else: print("\nNo valid predictions available to calculate overall accuracy.")
            else: print("\nNo metrics generated. Check processing steps and data.")
        else:
            print("\nNo reference column provided. Skipping accuracy calculation.")
            print(f"Processed data saved to '{processed_output_path}'. Metrics file '{output_path}' will not be created.")

        end_time = time.time()
        print(f"\n--- Processing complete in {end_time - start_time:.2f} seconds ---")
        return metric_df

    except FileNotFoundError: print(f"Error: Input file not found at '{input_path}'"); return None
    except ValueError as ve: print(f"Configuration Error: {ve}"); return None
    except Exception as e: print(f"An unexpected error occurred in main: {str(e)}"); traceback.print_exc(); return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model predictions for Multiple Choice Questions, extract answers, and optionally calculate accuracy.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add all arguments... (Ensure they match the previous version)
    parser.add_argument("--input", required=True, help="Path to input data file (CSV, TSV, TXT)")
    parser.add_argument("--output", required=True, help="Path to output file for metrics (CSV/TSV). Processed data saved separately.")
    parser.add_argument("--target", required=True, help="Column name containing raw model predictions/text.")
    parser.add_argument("--reference", default=None, help="Column name containing ground truth answers (required for accuracy calculation).")
    parser.add_argument("--input-format", choices=["csv", "tsv", "txt", "infer"], default=None, help="Format of input file. If None, attempts auto-detect.")
    parser.add_argument("--delimiter", default=None, help="Custom delimiter for input file (e.g., ',', 'tab', '|'). Overrides auto-detection.")
    parser.add_argument("--option-format", choices=["letter", "number"], default="letter", help="Format of options (e.g., 'a, b, c' or '1, 2, 3').")
    parser.add_argument("--option-range", choices=["a-d", "a-f", "1-4", "1-6"], default="a-d", help="Range of valid options.")
    parser.add_argument("--no-answer-token", default="x", help="Special token used by the LLM or expected in output to indicate no answer.")
    parser.add_argument("--handle-x", choices=["none", "keep", "random"], default="random", help="How to treat the no_answer_token ('x') during final label assignment.")
    parser.add_argument("--skip-regex", action="store_true", help="Skip the initial regex extraction step and rely solely on LLM extraction (requires --use-llm).")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM extraction via Ollama (as fallback or primary if --skip-regex).")
    parser.add_argument("--model", default="llama3:latest", help="Ollama model name to use for LLM extraction.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum context window size (in tokens) for the Ollama model.")
    parser.add_argument("--enable-cot", action="store_true", help="Enable Chain of Thought (CoT) prompting for LLM extraction.")
    parser.add_argument("--examples-file", default=None, help="Path to a file containing few-shot examples for LLM extraction.")

    args = parser.parse_args()

    # Process special delimiter values like 'tab'
    if args.delimiter == 'tab': args.delimiter = '\t'
    elif args.delimiter == 'pipe': args.delimiter = '|'

    # Run the main processing function, passing all relevant args
    main(input_path=args.input,
         output_path=args.output,
         target_col=args.target,
         reference_col=args.reference,
         use_llm=args.use_llm,
         llm_model=args.model,
         handle_x=args.handle_x,
         no_answer_token=args.no_answer_token,
         option_format=args.option_format,
         option_range=args.option_range,
         input_format=args.input_format,
         delimiter=args.delimiter,
         max_tokens=args.max_tokens,
         enable_cot=args.enable_cot,
         examples_path=args.examples_file,
         skip_regex=args.skip_regex) # Pass the parsed skip_regex value