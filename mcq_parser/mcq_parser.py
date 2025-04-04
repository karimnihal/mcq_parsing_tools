import re
import string
import random
import pandas as pd
import requests
import json
import time
import os
import argparse
import traceback
from sklearn.metrics import accuracy_score
from typing import Optional, Dict, Any, List, Tuple, Union

def extract_predicted_option(text: str, option_format="letter", option_range="a-d"):
    """
    Extract a predicted option from text using regex pattern matching.
    
    This function applies multiple regex strategies in order of reliability to find
    the most likely answer option in the provided text.
    
    Parameters:
        text (str): The text to extract an option from
        option_format (str): Either "letter" for a-f or "number" for 1-6
        option_range (str): The range of options, e.g. "a-d", "a-f", "1-4", "1-6"
    
    Returns:
        str or None: Extracted option or None if extraction failed
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

    # 1st extraction: option followed by a colon or period (e.g., "c:", "c.", "3:", "3.")
    pattern1 = f'(?<!\\w)([{escaped_options}])[:.]'
    match1 = re.search(pattern1, text_lower)
    if match1:
        return match1.group(1)

    # Look for phrases that strongly indicate an answer
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
    ]

    for pattern in answer_indicators:
        match = re.search(pattern, text_lower)
        if match:
            # Group index might vary based on pattern structure
            if len(match.groups()) == 1:
                return match.group(1)
            elif len(match.groups()) > 1:
                 # Handle patterns with multiple capture groups
                 return match.group(len(match.groups()))


    # 2nd extraction: standalone option character not preceded/followed by word chars
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
    Extract an answer option from text using a local LLM via Ollama.
    
    This function prepares a prompt for the LLM to extract a single token answer
    from the provided text. It can use few-shot examples and chain-of-thought reasoning.
    
    Parameters:
        text (str): The text to analyze
        model (str): Ollama model name to use
        no_answer_token (str): Token used to indicate no clear answer
        option_format (str): Either "letter" for a-f or "number" for 1-6
        option_range (str): The range of options, e.g. "a-d", "a-f", "1-4", "1-6" 
        max_tokens (int): Maximum context length for the model
        enable_cot (bool): Enable Chain-of-Thought reasoning in the prompt
        examples (list): Optional few-shot examples to include in the prompt
        
    Returns:
        str or None: Extracted option or None if extraction failed
    """
    if text is None:
        return None

    examples = examples or []

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

    # Token estimation and text truncation
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

    # Create the prompt
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

    # Validate response
    if response in valid_options_set and len(response) == 1:
        return response
    else:
        # Try regex as fallback
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
    Call a local LLM via the Ollama API.
    
    Parameters:
        prompt (str): The prompt to send to the model
        model (str): The Ollama model to use
        valid_options (str): String of valid option characters
        max_tokens (int): Maximum context length for the model
        enable_cot (bool): Whether Chain-of-Thought is enabled (affects output processing)
        
    Returns:
        str: The generated text response
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
    
    Parameters:
        model (str): The model name to check for
        
    Returns:
        bool: True if Ollama is available and the model is installed, False otherwise
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
    Normalize option text to a standard format.
    
    Parameters:
        text (str): The text to normalize
        option_format (str): Either "letter" for a-f or "number" for 1-6
        option_range (str): The range of options, e.g. "a-d", "a-f", "1-4", "1-6"
        no_answer_token (str): The token used for "no answer"
        
    Returns:
        str or None: Normalized option character or None if invalid
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
    Assign a final prediction label, handling special cases like 'x' and None values.
    
    Parameters:
        row (pd.Series): DataFrame row containing the data
        target_col (str): Column containing extracted prediction
        reference_col (str, optional): Column with reference answer for random assignment
        handle_x (str): How to handle the no_answer_token ('none', 'keep', 'random')
        option_format (str): Either "letter" for a-f or "number" for 1-6
        option_range (str): The range of options, e.g. "a-d", "a-f", "1-4", "1-6"
        no_answer_token (str): Token used to indicate no clear answer
        
    Returns:
        str or None: Final assigned label
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
    
    Parameters:
        df (pd.DataFrame): DataFrame containing predictions and references
        target_col (str): Column containing predictions
        reference_col (str): Column containing reference answers
        
    Returns:
        pd.DataFrame: DataFrame with accuracy metrics by language and prompt
    """
    results = []
    df[reference_col] = df[reference_col].astype(str).str.lower()
    valid_mask = df[target_col].notna() & (df[target_col] != 'x')
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
    """
    Load few-shot examples from a file.
    
    The file should contain examples in the format: "text:::option"
    
    Parameters:
        examples_path (str, optional): Path to the examples file
        
    Returns:
        list or None: List of dictionaries with 'text' and 'option' keys, or None if loading failed
    """
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

def process_predictions(df, target_col, reference_col, use_llm=False, llm_model="llama3:latest",
                        handle_x="random", no_answer_token="x", option_format="letter", option_range="a-d",
                        max_tokens=4096, enable_cot=False, examples=None, skip_regex: bool = False):
    """
    Process predictions using extraction methods, verbalizer, and label assignment.
    
    This function implements a pipeline for processing MCQ responses:
    1. Extract answer options using regex pattern matching (unless skip_regex=True)
    2. Use LLM to extract answers where regex failed (if use_llm=True)
    3. Normalize extracted options with verbalizer
    4. Assign final labels with configurable handling of special cases
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data to process
        target_col (str): Column containing raw predictions
        reference_col (str, optional): Column with reference answers
        use_llm (bool): Whether to use LLM extraction
        llm_model (str): Ollama model to use for LLM extraction
        handle_x (str): How to handle the no_answer_token ('none', 'keep', 'random')
        no_answer_token (str): Token used to indicate no clear answer
        option_format (str): Either "letter" for a-f or "number" for 1-6
        option_range (str): The range of options, e.g. "a-d", "a-f", "1-4", "1-6"
        max_tokens (int): Maximum context length for the LLM
        enable_cot (bool): Enable Chain-of-Thought reasoning for LLM
        examples (list): Few-shot examples for LLM extraction
        skip_regex (bool): Skip the regex extraction step
        
    Returns:
        pd.DataFrame: Processed DataFrame with extracted options and final labels
    """
    processed_df = df.copy()
    raw_prediction_col = target_col + '_raw'
    processed_df[raw_prediction_col] = processed_df[target_col]

    # Initialize extracted_option column
    processed_df['extracted_option'] = pd.Series(dtype='object')
    processed_df['extracted_option'] = pd.NA

    # Step 1: Apply regex extraction (if not skipped)
    if not skip_regex:
        print("Applying regex extraction...")
        extracted_options_series = processed_df[raw_prediction_col].apply(
            lambda text: extract_predicted_option(text, option_format, option_range)
        )
        processed_df.loc[extracted_options_series.notna(), 'extracted_option'] = extracted_options_series.dropna()
        print(f"Regex extraction complete. Found options for {processed_df['extracted_option'].notna().sum()}/{len(processed_df)} entries.")
    else:
        print("Skipping regex extraction as requested.")

    # Step 2: Apply LLM extraction if enabled
    llm_applied_count = 0
    if use_llm:
        ollama_available = check_ollama_availability(model=llm_model)
        if ollama_available:
            # Determine which rows need LLM processing
            if skip_regex:
                mask_needs_llm = pd.Series(True, index=processed_df.index)
                print(f"Using LLM '{llm_model}' for all {len(processed_df)} entries (regex skipped)...")
            else:
                mask_needs_llm = processed_df['extracted_option'].isna()
                num_to_process_with_llm = mask_needs_llm.sum()
                if num_to_process_with_llm > 0:
                    print(f"Using LLM '{llm_model}' for {num_to_process_with_llm} entries where regex extraction failed...")

            if mask_needs_llm.any():
                if examples:
                    print(f"Using {len(examples)} few-shot examples.")

                def apply_llm_extraction_for_row(row_data):
                    print(f"  LLM Processing text starting with: '{row_data[raw_prediction_col][:50]}...'")
                    return extract_predicted_option_llm(
                        row_data[raw_prediction_col],
                        model=llm_model,
                        no_answer_token=no_answer_token,
                        option_format=option_format,
                        option_range=option_range,
                        max_tokens=max_tokens,
                        enable_cot=enable_cot,
                        examples=examples
                    )

                rows_to_process = processed_df.loc[mask_needs_llm]
                print(f"Applying LLM extraction to {len(rows_to_process)} rows...")
                llm_results = rows_to_process.apply(apply_llm_extraction_for_row, axis=1)
                llm_applied_count = len(llm_results)

                processed_df.loc[mask_needs_llm, 'extracted_option'] = llm_results.fillna(pd.NA)

                print(f"LLM extraction applied to {llm_applied_count} entries. "
                      f"Total options found (Regex+LLM): {processed_df['extracted_option'].notna().sum()}/{len(processed_df)}")
            else:
                if not skip_regex:
                     print("Regex extraction succeeded for all entries. Skipping LLM.")

        else:
            print(f"Warning: Ollama API not available or model '{llm_model}' not installed. Cannot use LLM.")
            if skip_regex:
                print("Warning: Regex was skipped and LLM is unavailable. No extraction will be performed.")

    # Step 3: Apply verbalizer
    print("Applying verbalizer for normalization...")
    processed_df['verbalized_option'] = processed_df['extracted_option'].apply(
        lambda text: verbalizer(text, option_format, option_range, no_answer_token)
    ).fillna(pd.NA)
    print(f"Verbalizer complete. Valid options/tokens found for {processed_df['verbalized_option'].notna().sum()}/{len(processed_df)} entries.")

    # Step 4: Assign final label
    print(f"Assigning final labels (handle_x='{handle_x}')...")
    final_label_col = target_col
    processed_df[final_label_col] = processed_df.apply(
        lambda row: assign_label(row, 'verbalized_option', reference_col, handle_x, option_format, option_range, no_answer_token),
        axis=1
    )
    print("Final label assignment complete.")

    return processed_df

def main(input_path: str, output_path: str, target_col: str, reference_col: Optional[str],
         use_llm: bool = False, llm_model: str = "llama3:latest", handle_x: str = "random",
         no_answer_token: str = "x", option_format: str = "letter", option_range: str = "a-d",
         input_format: Optional[str] = None, delimiter: Optional[str] = None, max_tokens: int = 4096,
         enable_cot: bool = False, examples_path: Optional[str] = None,
         skip_regex: bool = False):
    """
    Main function to process MCQ predictions and calculate accuracy metrics.
    
    This function orchestrates the full pipeline:
    1. Read and validate input data
    2. Process predictions using regex and/or LLM extraction
    3. Calculate accuracy metrics if reference answers are provided
    4. Save results to output files
    
    Parameters:
        input_path (str): Path to input data file
        output_path (str): Path to output metrics file
        target_col (str): Column containing raw predictions
        reference_col (str, optional): Column with reference answers
        use_llm (bool): Whether to use LLM extraction
        llm_model (str): Ollama model to use for LLM extraction
        handle_x (str): How to handle the no_answer_token
        no_answer_token (str): Token used to indicate no clear answer
        option_format (str): Option format (letter or number)
        option_range (str): Range of valid options
        input_format (str, optional): Format of input file
        delimiter (str, optional): Custom delimiter for input file
        max_tokens (int): Maximum context length for the LLM
        enable_cot (bool): Enable Chain-of-Thought for LLM
        examples_path (str, optional): Path to few-shot examples file
        skip_regex (bool): Skip the regex extraction step
        
    Returns:
        pd.DataFrame or None: DataFrame with accuracy metrics, or None if processing failed
    """
    print("--- Starting MCQ Parser ---")
    print(f"  main received: skip_regex={skip_regex}, use_llm={use_llm}, enable_cot={enable_cot}, handle_x={handle_x}")
    try:
        # File Reading
        start_time = time.time()
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

        # Column Validation
        required_cols = {target_col, 'language', 'prompt_no'}
        if reference_col: required_cols.add(reference_col)
        missing_cols = required_cols - set(df.columns)
        if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")

        # Load Examples
        examples = load_examples(examples_path)

        # Prepare Columns
        df[target_col] = df[target_col].fillna('').astype(str)
        if reference_col: df[reference_col] = df[reference_col].fillna('').astype(str).str.lower()
        df['language'] = df['language'].astype(str)
        df['prompt_no'] = df['prompt_no'].astype(str)

        # Process Predictions
        print("\n--- Processing Predictions ---")
        print(f"Target Column: '{target_col}'")
        if reference_col: print(f"Reference Column: '{reference_col}'")
        print(f"Option Format: {option_format}, Option Range: {option_range}")
        print(f"Skip Regex: {skip_regex}")
        print(f"Use LLM: {use_llm}")
        if use_llm:
            print(f"LLM Model: {llm_model}, Max Tokens: {max_tokens}, CoT: {enable_cot}")
            if examples: print(f"Few-shot Examples: {len(examples)} loaded")
        print(f"Handle '{no_answer_token}' Token: {handle_x}")

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
            skip_regex=skip_regex
        )

        # Determine Output Format & Save
        output_ext = os.path.splitext(output_path)[1].lower()
        output_format = 'tsv' if output_ext == '.tsv' else 'csv'
        output_delim = '\t' if output_format == 'tsv' else ','

        processed_output_path = output_path.replace(output_ext, f"_processed_data{output_ext}")
        print(f"\nSaving detailed processed data to: '{processed_output_path}'")
        processed_df.to_csv(processed_output_path, sep=output_delim, index=False, encoding='utf-8')

        # Calculate Metrics (if reference column provided)
        metric_df = None
        if reference_col:
            print("\n--- Calculating Accuracy Metrics ---")
            metric_df = calculate_metric(processed_df, target_col, reference_col)
            if metric_df is not None and not metric_df.empty:
                print(f"Saving accuracy metrics to: '{output_path}'")
                metric_df.to_csv(output_path, sep=output_delim, index=False, encoding='utf-8')
                
                # Print Summary
                print("\n--- Accuracy Summary ---")
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