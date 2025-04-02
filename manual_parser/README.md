# Manual MCQ Answer Parser

A command-line tool for interactively reviewing text content (e.g., language model outputs) and manually assigning or correcting multiple-choice question (MCQ) answers.

## Overview

This script provides a simple interface to step through rows of data, view specific text content, and input the correct MCQ answer (e.g., 'a', 'b', 'c', 'd', 'x'). It reads source data from an input file and reads/writes the manually assigned answers to a separate output file, allowing you to iteratively build or refine a dataset of parsed answers. It supports resuming interrupted sessions and can handle data with or without a dedicated index column.

## Features

-   **Interactive Interface:** Simple command-line display for reviewing data row by row.
-   **Context Display:** Shows content from a designated source column (e.g., model output) for review.
-   **Shows Existing Answers:** Displays the current value in the target parsed answer column, if any.
-   **Flexible Input:** Accepts standard MCQ options (configurable, e.g., a-d, 1-6) and a 'no answer' token (e.g., 'x').
-   **Configurable Options:** Allows specifying option format (letter/number), range (a-d, 1-6, etc.), and the token used for 'no answer'.
-   **Handles Various File Formats:** Supports TSV, CSV, or TXT input/output with auto-detection or explicit delimiter setting.
-   **Resumes Sessions:** Detects the first row without a valid answer in the output file and offers to resume from that point.
-   **Indexed or Sequential Processing:**
    *   Aligns input data and output answers using a specified unique index column (`--index-col`) for reliable merging.
    *   Supports processing files without a dedicated index column sequentially (`--no-index`), relying on row order (use with caution).

## Installation

The script requires Python 3 and the pandas library.

```bash
# Install pandas if you haven't already
pip install pandas
```

## Usage

**Basic usage (TSV files, default index `Unnamed: 0`):**

```bash
python manual_parser.py \
    --input sample.tsv \
    --output sample_manual_parsed.tsv \
    --source-col model_output_text \
    --parsed-col manual_label \
    --index-col "Unnamed: 0"
```

**Using a named index column (`question_id`):**

```bash
python manual_parser.py \
    --input raw_data.tsv \
    --output parsed_answers.tsv \
    --source-col model_output_text \
    --parsed-col manual_label \
    --index-col question_id
```

**Using CSV files:**

```bash
python manual_parser.py \
    --input raw_data.csv \
    --output parsed_answers.csv \
    --source-col gpt_response \
    --parsed-col verified_answer \
    --index-col unique_id \
    --delimiter ","
```

**Processing files sequentially (no index column):**

*   **Note:** Use with caution. Assumes input and output files (if output exists) have perfectly matching row order. Mismatched row counts will cause existing answers to be discarded. Best for starting fresh or when certain of alignment.*

```bash
python manual_parser.py \
    --input data_no_index.tsv \
    --output results_no_index.tsv \
    --source-col text_to_parse \
    --parsed-col assigned_label \
    --no-index
```

**Using different option formats (e.g., numeric 1-6, 'na' for no answer):**

```bash
python manual_parser.py \
    --input raw_data.tsv \
    --output parsed_answers.tsv \
    --source-col response \
    --parsed-col numeric_label \
    --index-col sample_id \
    --option-format number \
    --option-range 1-6 \
    --no-answer-token "na"
```

## Arguments

| Argument            | Description                                                                                                                   | Default        | Required         |
| :------------------ | :---------------------------------------------------------------------------------------------------------------------------- | :------------- | :--------------- |
| `--input`           | Path to input data file (CSV, TSV, TXT) containing source text and index (if used).                                            | -              | Yes              |
| `--output`          | Path to output file (CSV, TSV) where manual parsed answers are stored/updated. Will be created if it doesn't exist.             | -              | Yes              |
| `--source-col`      | Column name in the `--input` file containing the text to display for manual parsing.                                            | -              | Yes              |
| `--parsed-col`      | Column name in the `--output` file to store/update the manually entered parsed answer.                                          | -              | Yes              |
| `--index-col`       | Column name for aligning rows (must be unique & in both files if `--output` exists). Use 'Unnamed: 0' for pandas default.       | `None`         | Yes (unless `--no-index`) |
| `--no-index`        | Indicate input file has NO dedicated index column. Process sequentially. Cannot reliably merge if row counts differ.            | `False`        | No               |
| `--input-format`    | Format of input file: `csv`, `tsv`, `txt`, or `infer`. If `None`, attempts auto-detect from extension.                        | `None`         | No               |
| `--delimiter`       | Custom delimiter for input/output files (e.g., `,`, `tab`, `\|`). If `None`, inferred from format/extension.                     | `None`         | No               |
| `--option-format`   | Expected format of options ('letter' or 'number'). Determines valid inputs.                                                   | `letter`       | No               |
| `--option-range`    | Expected range of options (e.g., 'a-d', '1-6'). Determines valid inputs.                                                       | `a-d`          | No               |
| `--no-answer-token` | Special token allowed as input to indicate no valid answer could be determined manually.                                        | `x`            | No               |

## Workflow

1.  **Load Input:** Reads the `--input` file based on the specified or inferred format/delimiter.
2.  **Load/Check Output:** Checks if the `--output` file exists.
    *   If it exists, it attempts to load it.
    *   If using `--index-col`, it validates the presence of the index and parsed columns. It then merges the existing parsed answers onto the input data based on the shared index column. Missing parsed columns are added.
    *   If using `--no-index`, it checks for row count mismatches. If counts differ or loading fails, it proceeds as if the output file didn't exist (starting fresh). Otherwise, it loads answers positionally.
    *   If the output file *doesn't* exist, it prepares to create it by adding the `--parsed-col` to the input data structure.
3.  **Resume Prompt:** Checks the (potentially merged) data for the first row where the `--parsed-col` does not contain a valid answer (according to `--option-format`, `--option-range`, `--no-answer-token`). It asks the user if they want to resume from this point or start/review from the beginning.
4.  **Interactive Loop:**
    *   For each row (starting from the beginning or the resume point):
        *   Clears the screen.
        *   Displays the row number and the value from the index column (if used).
        *   Displays the text from the `--source-col`.
        *   Displays the current value in the `--parsed-col`.
        *   Prompts the user to enter a valid answer (`a/b/c/d/x`, etc.), skip (`s`), or quit (`q`).
        *   Updates the parsed answer in memory based on user input.
5.  **Save Output:** When the user quits (`q`) or finishes all rows, the script saves the *entire* dataset (original input columns + updated `--parsed-col`) to the `--output` file, overwriting the previous version.

## Indexing (`--index-col` vs. `--no-index`)

-   **`--index-col` (Recommended):** Use this when your input data has a column with unique identifiers (e.g., `question_id`, `sample_id`, or the default `Unnamed: 0` if your TSV starts with `\t`). This allows the script to reliably match rows between your `input` data and any existing `output` file, even if the row order differs or rows are added/removed in the input. The script merges existing answers based on this ID. **The index column values must be unique.**
-   **`--no-index`:** Use this *only* if your data has no unique identifier column. The script processes rows sequentially based purely on their order in the file.
    *   **Warning:** If you use `--no-index` and the `--output` file exists from a previous run, the script *assumes* the row order is identical to the current `--input` file. If rows have been added, removed, or reordered in the input since the output was last saved, merging existing answers will be incorrect. The script warns about row count mismatches and will effectively start fresh in that case.

## Output File

The script generates or updates **one** file specified by the `--output` argument. This file contains:

*   All columns from the original `--input` file.
*   The `--parsed-col` containing the manually assigned answers (or existing answers that were skipped).

The format (CSV/TSV) is typically inferred from the output file extension, and the delimiter matches the input unless specified otherwise.