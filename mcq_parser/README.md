# MCQ Parser

A tool for extracting multiple-choice answers from language model outputs, optionally using few-shot examples for improved LLM performance, and calculating accuracy metrics. Offers flexibility by allowing regex-only, LLM-only, or combined extraction strategies.

## Overview

MCQ Parser is designed to extract and analyze answers in multiple-choice question formats from language model responses. It combines robust regex-based pattern matching with optional LLM-powered extraction (using local Ollama models) for challenging cases. The script now supports providing few-shot examples to guide the LLM for more accurate and context-aware extraction, and **can optionally skip the regex step entirely** to rely solely on the LLM. It handles various answer formats and file types.

## Features

-   Robust regex-based answer extraction.
-   Optional LLM fallback using local Ollama models.
-   Support for few-shot examples to improve LLM extraction accuracy.
-   Option to skip regex extraction and use only LLM (`--skip-regex`).
-   Support for different option formats (letters a-d, a-f; numbers 1-4, 1-6).
-   Flexible input and output file handling (CSV, TSV, TXT with auto-detection).
-   Optional Chain of Thought (CoT) reasoning for enhanced LLM extraction.
-   Detailed accuracy metrics and reporting (if reference answers are provided).
-   Configurable handling of ambiguous or missing answers.

## Installation

This script requires Python 3, pandas, scikit-learn, and requests.

### Ollama Setup (Optional but Required for LLM Features)

To use the LLM fallback, few-shot examples, or the `--skip-regex` feature:

1.  Install Ollama: [https://ollama.ai/](https://ollama.ai/)
2.  Pull your preferred model(s) via the command line (e.g., `llama3`, `mistral`, `deepseek-coder`):
    ```bash
    ollama pull llama3:latest
    ollama pull mistral:latest
    ```
3.  Ensure the Ollama server is running (it usually starts automatically after installation).

## Usage

**Basic usage (Regex only, default):**

```bash
python mcq_parser.py \
    --input "data.tsv" \
    --output "results.csv" \
    --target "model_output" \
    --reference "correct_answer"
```

**Using LLM Fallback (Regex first, then LLM if regex fails):**

```bash
python mcq_parser.py \
    --input "data.tsv" \
    --output "results.csv" \
    --target "model_output" \
    --reference "correct_answer" \
    --use-llm \
    --model "llama3:latest"
```

**Using LLM Only (Skipping Regex):**

```bash
python mcq_parser.py \
    --input "data.tsv" \
    --output "results.csv" \
    --target "model_output" \
    --reference "correct_answer" \
    --use-llm \
    --skip-regex \
    --model "mistral:latest"
```

**Using LLM Only with Few-Shot Examples and CoT:**

```bash
python mcq_parser.py \
    --input "data.tsv" \
    --output "results.csv" \
    --target "model_output" \
    --reference "correct_answer" \
    --use-llm \
    --skip-regex \
    --model "mistral:latest" \
    --enable-cot \
    --examples-file "my_mcq_examples.txt"
```

## Arguments

| Argument            | Description                                                                                                                   | Default        | Required         |
| :------------------ | :---------------------------------------------------------------------------------------------------------------------------- | :------------- | :--------------- |
| `--input`           | Path to input data file (CSV, TSV, TXT).                                                                                      | -              | Yes              |
| `--output`          | Path to output file for metrics (CSV/TSV). Processed data saved separately.                                                   | -              | Yes              |
| `--target`          | Column name containing raw model predictions/text.                                                                            | -              | Yes              |
| `--reference`       | Column name containing ground truth answers (required for accuracy calculation).                                                | `None`         | No (for metrics) |
| `--input-format`    | Format of input file: `csv`, `tsv`, `txt`, or `infer`. If `None`, attempts auto-detect from extension.                        | `None`         | No               |
| `--delimiter`       | Custom delimiter for input file (e.g., `,`, `tab`, `\|`). Overrides auto-detection.                                             | `None`         | No               |
| `--option-format`   | Format of options: `letter` or `number`.                                                                                      | `letter`       | No               |
| `--option-range`    | Range of options: `a-d`, `a-f`, `1-4`, or `1-6`.                                                                              | `a-d`          | No               |
| `--no-answer-token` | Special token to indicate no answer (e.g., from LLM).                                                                         | `x`            | No               |
| `--handle-x`        | How to treat `no_answer_token`: `none`, `keep`, or `random`. See description below.                                             | `random`       | No               |
| `--use-llm`         | Enable LLM extraction via Ollama (either as fallback or primary method if `--skip-regex` is used).                             | `False`        | No               |
| **`--skip-regex`**  | **NEW:** Skip the initial regex extraction step. Requires `--use-llm` to perform any extraction.                                | `False`        | No               |
| `--model`           | Ollama model name to use for LLM extraction (e.g., `llama3:latest`, `mistral:7b`).                                             | `llama3:latest`| No               |
| `--max-tokens`      | Maximum context window size (in tokens) for the Ollama model. Affects truncation with examples.                                 | `4096`         | No               |
| `--enable-cot`      | Enable Chain of Thought (CoT) prompting for LLM extraction (encourages reasoning steps).                                        | `False`        | No               |
| `--examples-file`   | Path to a file containing few-shot examples for LLM extraction. Format: `Input Text ::: expected_option`.                        | `None`         | No               |

**Details on `--handle-x`:**

*   `none`: If the extracted/verbalized answer is the `no_answer_token` (e.g., 'x'), the final label becomes `None`. These entries are **excluded** from accuracy calculations.
*   `keep`: The `no_answer_token` is kept as the final label. These entries are **excluded** from accuracy calculations (unless the reference answer is also 'x', which is unlikely).
*   `random`: If the answer is the `no_answer_token` or invalid, assign a random *incorrect* standard label (e.g., if the correct answer is 'a', assign 'b', 'c', or 'd' randomly). These entries are **included** in the accuracy calculation denominator, effectively penalizing non-answers.

## Using Few-Shot Examples (`--examples-file`)

Providing few-shot examples can significantly improve the LLM's ability to extract the correct answer option, especially when the model's output format is inconsistent or includes extensive reasoning before the final choice.

**Example File Format (`my_mcq_examples.txt`):**

Create a plain text file where each line represents one example, following the format:

```text
Input Text From Model Output ::: desired_single_option_character
```

*   Use `:::` as the separator.
*   Whitespace around the text and option is trimmed.
*   The `desired_single_option_character` should be the exact character you want extracted (e.g., `a`, `b`, `1`, `2`, `x`).
*   Lines starting with `#` are ignored as comments.
*   Empty lines are ignored.

**Example Content:**

```text
# Examples for letter options (a-d)
The best choice appears to be C. After reviewing the evidence, C aligns most closely. ::: c
Based on the text, the answer is A because of the first paragraph. ::: a
I will select option (B) as it is the only logical conclusion. ::: b
Final Answer: D ::: d
I think the answer is a). ::: a
Hmm, none of the options seem correct. ::: x

# Examples for number options (1-4)
It's likely 2 based on the calculation. ::: 2
The correct number is 1. See line 5. ::: 1
I am unable to determine the answer from the provided text. ::: x
Result: 4 ::: 4
```

**Token Considerations:**

*   The examples are added to the prompt sent to the LLM.
*   The script estimates the token count of the examples and the base prompt instructions.
*   The main input text (`--target` column content) will be **truncated** if the combined size (examples + prompt + input text) exceeds the `--max-tokens` limit specified for the model. A warning will be printed if truncation occurs.
*   Ensure your `--max-tokens` setting is appropriate for the chosen Ollama model and accommodates both your examples and typical input text length.

## Working with Ollama Models

The parser supports using Ollama models for extraction, either as a fallback or as the sole method.

1.  **Extraction Strategy**:
    *   Default: Regex first.
    *   `--use-llm`: Regex first, then LLM for entries where regex failed.
    *   `--use-llm --skip-regex`: LLM only for all entries.
2.  **Model Selection**: Choose a model installed in your Ollama setup (use `ollama list` to see available models). Different models have varying context sizes (`--max-tokens`), speeds, and instruction-following capabilities. Consider models fine-tuned for instruction following (like many `Mistral`, `Llama`, `DeepSeek`, or `Phi` variants).
3.  **Context Size (`--max-tokens`)**: Set this according to your model's limit (e.g., 4096, 8192, 32768). Remember that few-shot examples consume context window space.
4.  **Chain of Thought (`--enable-cot`)**: Recommended for complex extraction tasks or when the model output includes reasoning. CoT prompts the model to "think" step-by-step (using `<think>` tags internally) before giving the final answer, often improving accuracy for capable models. It requires slightly more response tokens from the model.
5.  **Few-Shot Examples (`--examples-file`)**: Provide examples to guide the model, especially useful if the model's default output style doesn't cleanly present the answer or if you need it to handle specific edge cases (like identifying when no answer is given).
6.  **Performance**: Larger models might be more accurate but slower. Using LLM extraction (especially with `--skip-regex`, many examples, or long input texts) will be significantly slower than regex-only processing.

## Output

The script produces up to two files:

1.  **Metrics File** (path specified by `--output`): Contains accuracy metrics grouped by `language` and `prompt_no`. This file is only created if a `--reference` column is provided and valid predictions are available for comparison. Columns include:
    *   `language`, `prompt_no`
    *   `accuracy`: Percentage accuracy for the group.
    *   `correct`: Number of correct predictions.
    *   `count_in_calc`: Number of predictions used in the accuracy calculation for this group (excludes entries handled as `None` or `keep='x'`).
    *   `total_original`: Total number of original rows for this group in the input file.
2.  **Processed Data File** (`{output_base}_processed_data.{ext}`): Always created. Contains the original data plus additional columns showing the results of the parsing steps:
    *   `{target_col}_raw`: The original input text from the target column.
    *   `extracted_option`: The option found by regex or LLM *before* normalization (can be NA if extraction failed).
    *   `verbalized_option`: The normalized single character option (or `no_answer_token`) after cleaning (can be NA).
    *   `{target_col}`: The **final assigned label** after applying `handle_x` rules. This is the column compared against the reference for accuracy.

---

## Appendix: Core Function Reference

*   **`extract_predicted_option`**: Extracts answers using regex patterns.
*   **`extract_predicted_option_llm`**: Uses an LLM (via Ollama) to extract answers, potentially using few-shot examples and considering token limits.
*   **`call_ollama_api`**: Makes API calls to the local Ollama server.
*   **`load_examples`**: Loads and parses the few-shot examples file.
*   **`check_ollama_availability`**: Verifies Ollama connection and model availability.
*   **`verbalizer`**: Normalizes extracted text to a single valid option character or `no_answer_token`.
*   **`assign_label`**: Applies rules (`handle_x`) to determine the final label based on the verbalized option.
*   **`calculate_metric`**: Computes accuracy metrics grouped by categories.
*   **`process_predictions`**: Manages the complete extraction pipeline (optional regex -> optional LLM -> verbalizer -> assign_label).
*   **`main`**: Handles command-line arguments, file I/O, and orchestrates the overall process.
