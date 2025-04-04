import pytest
import pandas as pd
import numpy as np
import random
import os
import json
from unittest.mock import patch, MagicMock, call

# Import modules for testing
import mcq_parser
import requests

# ============================================================================
# Tests for extract_predicted_option
# ============================================================================

@pytest.mark.parametrize(
    "text, option_format, option_range, expected_output",
    [
        # Basic Letter Cases (a-d)
        ("Answer: a", "letter", "a-d", "a"),
        ("The choice is b.", "letter", "a-d", "b"),
        ("I select c:", "letter", "a-d", "c"),
        ("option d is correct", "letter", "a-d", "d"),
        ("a: Explanation", "letter", "a-d", "a"),
        ("b. Because...", "letter", "a-d", "b"),
        ("This is c", "letter", "a-d", 'c'),
        ("The option is a", "letter", "a-d", "a"),
        ("correct answer is b", "letter", "a-d", "b"),
        ("b is correct", "letter", "a-d", "b"),
        ("final answer: c", "letter", "a-d", "c"),
        ("selected d", "letter", "a-d", "d"),
        ("chose a", "letter", "a-d", "a"),

        # Basic Letter Cases (a-f)
        ("Answer: e", "letter", "a-f", "e"),
        ("The choice is f.", "letter", "a-f", "f"),
        ("e: Explanation", "letter", "a-f", "e"),

        # Basic Number Cases (1-4)
        ("Answer: 1", "number", "1-4", "1"),
        ("The choice is 2.", "number", "1-4", "2"),
        ("I select 3:", "number", "1-4", "3"),
        ("option 4 is correct", "number", "1-4", "4"),
        ("1: Explanation", "number", "1-4", "1"),
        ("2. Because...", "number", "1-4", "2"),

        # Basic Number Cases (1-6)
        ("Answer: 5", "number", "1-6", "5"),
        ("The choice is 6.", "number", "1-6", "6"),
        ("5: Explanation", "number", "1-6", "5"),

        # Standalone option
        ("a", "letter", "a-d", "a"),
        ("  b  ", "letter", "a-d", "b"),
        ("1", "number", "1-4", "1"),
        ("  2  ", "number", "1-4", "2"),

        # Case Insensitivity
        ("ANSWER IS B", "letter", "a-d", "b"),
        ("Option C:", "letter", "a-d", "c"),

        # Priority (Colon/Period should often win over standalone/less specific indicators)
        ("d) is correct. Final answer: c:", "letter", "a-d", "c"),
        ("the answer is b. This is because...", "letter", "a-d", "b"),
        ("The a option vs the answer: b", "letter", "a-d", "b"),

        # Edge Cases & No Match
        ("(a)", "letter", "a-d", "a"),
        ("[b].", "letter", "a-d", "b"),
        ("'c':", "letter", "a-d", "c"),
        ("This text has no valid option.", "letter", "a-d", None),
        ("tablecloth", "letter", "a-d", None),
        ("1234", "number", "1-4", None),
        ("Answer is 5", "number", "1-4", None),
        ("Answer is g", "letter", "a-f", None),
        (None, "letter", "a-d", None),
        ("", "letter", "a-d", None),
        (" ", "letter", "a-d", None),
        ("Maybe a", "letter", "a-d", "a"),
        ("The answer could be b", "letter", "a-d", "b"),
    ]
)
def test_extract_predicted_option(text, option_format, option_range, expected_output):
    """
    Test the regex extraction logic for various formats and edge cases.
    
    This test verifies that the extract_predicted_option function correctly
    identifies MCQ answers in different text formats and handles edge cases.
    """
    assert mcq_parser.extract_predicted_option(text, option_format, option_range) == expected_output


# ============================================================================
# Tests for verbalizer
# ============================================================================

@pytest.mark.parametrize(
    "text, option_format, option_range, no_answer_token, expected_output",
    [
        # Basic Valid Cases
        ("a", "letter", "a-d", "x", "a"),
        ("1", "number", "1-4", "x", "1"),
        ("x", "letter", "a-d", "x", "x"),
        ("choose c", "letter", "a-d", "x", "c"),
        ("it is 2", "number", "1-4", "x", "2"),
        ("maybe x?", "letter", "a-d", "x", "x"),

        # Punctuation and Case
        ("b.", "letter", "a-d", "x", "b"),
        ("(c)", "letter", "a-d", "x", "c"),
        (" D ", "letter", "a-d", "x", "d"),
        (" X ", "letter", "a-d", "x", "x"),
        ("4!", "number", "1-4", "x", "4"),

        # First Option Priority
        ("a or b", "letter", "a-d", "x", "a"),
        ("1, maybe 2?", "number", "1-4", "x", "1"),
        ("Answer x or c", "letter", "a-d", "x", "x"),

        # No Valid Option
        ("Invalid choice", "letter", "a-d", "x", None),
        ("5", "number", "1-4", "x", None),
        ("g", "letter", "a-f", "x", None),
        ("xyz", "letter", "a-d", "x", None),

        # None/Empty Input
        (None, "letter", "a-d", "x", None),
        ("", "letter", "a-d", "x", None),
        ("   ", "letter", "a-d", "x", None),
        (".,!?", "letter", "a-d", "x", None),

        # Different Ranges/Formats
        ("e", "letter", "a-f", "x", "e"),
        ("6", "number", "1-6", "x", "6"),

        # Custom No Answer Token
        ("n/a", "letter", "a-d", "n/a", "n/a"),
        ("No answer provided.", "letter", "a-d", "n/a", None),
        ("The answer is n/a", "letter", "a-d", "n/a", None),
    ]
)
def test_verbalizer(text, option_format, option_range, no_answer_token, expected_output):
    """
    Test the normalization logic of the verbalizer function.
    
    This test ensures the verbalizer correctly normalizes extracted text
    into standardized option formats, handling special tokens and edge cases.
    """
    assert mcq_parser.verbalizer(text, option_format, option_range, no_answer_token) == expected_output


# ============================================================================
# Tests for assign_label
# ============================================================================

# Helper function to check random results
def check_random_output(output, possible_set):
    """Check if a randomly assigned output is within the expected set of possibilities."""
    print(f"Checking if '{output}' is in {possible_set}")
    return output in possible_set

@pytest.mark.parametrize(
    "row_dict, target_col, reference_col, handle_x, option_format, option_range, no_answer_token, expected_output",
    [
        # Basic Valid Predictions
        ({'pred': 'a', 'ref': 'a'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', 'a'),
        ({'pred': 'b', 'ref': 'a'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', 'b'),

        # Handling 'x' (no_answer_token)
        ({'pred': 'x', 'ref': 'a'}, 'pred', 'ref', 'none', 'letter', 'a-d', 'x', None),
        ({'pred': 'x', 'ref': 'a'}, 'pred', 'ref', 'keep', 'letter', 'a-d', 'x', 'x'),
        ({'pred': 'x', 'ref': 'a'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', {'b', 'c', 'd'}),
        ({'pred': 'x', 'ref': 'd'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', {'a', 'b', 'c'}),
        ({'pred': 'x', 'ref': '1'}, 'pred', 'ref', 'random', 'number', '1-4', 'x', {'2', '3', '4'}),

        # Handling None/Invalid Predictions
        ({'pred': None, 'ref': 'b'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', {'a', 'c', 'd'}),
        ({'pred': 'z', 'ref': 'c'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', {'a', 'b', 'd'}),
        ({'pred': None, 'ref': 'b'}, 'pred', 'ref', 'none', 'letter', 'a-d', 'x', {'a', 'c', 'd'}),
        ({'pred': '', 'ref': 'd'}, 'pred', 'ref', 'keep', 'letter', 'a-d', 'x', {'a', 'b', 'c'}),

        # Random assignment when reference is missing or invalid
        ({'pred': 'x'}, 'pred', None, 'random', 'letter', 'a-d', 'x', {'a', 'b', 'c', 'd'}),
        ({'pred': 'x', 'ref': None}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', {'a', 'b', 'c', 'd'}),
        ({'pred': 'x', 'ref': 'z'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'x', {'a', 'b', 'c', 'd'}),

        # Different Ranges/Formats with Random
        ({'pred': 'x', 'ref': 'f'}, 'pred', 'ref', 'random', 'letter', 'a-f', 'x', {'a', 'b', 'c', 'd', 'e'}),
        ({'pred': None, 'ref': '6'}, 'pred', 'ref', 'random', 'number', '1-6', 'x', {'1', '2', '3', '4', '5'}),

        # Custom No Answer Token
        ({'pred': 'n/a', 'ref': 'a'}, 'pred', 'ref', 'keep', 'letter', 'a-d', 'n/a', 'n/a'),
        ({'pred': 'n/a', 'ref': 'a'}, 'pred', 'ref', 'none', 'letter', 'a-d', 'n/a', None),
        ({'pred': 'n/a', 'ref': 'a'}, 'pred', 'ref', 'random', 'letter', 'a-d', 'n/a', {'b', 'c', 'd'}),
    ]
)
def test_assign_label(row_dict, target_col, reference_col, handle_x, option_format, option_range, no_answer_token, expected_output):
    """
    Test the final label assignment logic.
    
    This test verifies that the assign_label function correctly handles different
    scenarios for final prediction assignment, including the special treatment of
    the no_answer_token based on the handle_x parameter.
    """
    # Simulate a DataFrame row using a Series
    row = pd.Series(row_dict)

    actual_output = mcq_parser.assign_label(
        row, target_col, reference_col, handle_x, option_format, option_range, no_answer_token
    )

    if isinstance(expected_output, set):
        # Check if the random output is one of the expected possibilities
        assert actual_output in expected_output
    else:
        # Check for exact match for non-random cases
        assert actual_output == expected_output


# ============================================================================
# Tests for load_examples
# ============================================================================

def test_load_examples_valid_file(tmp_path):
    """
    Test loading a valid examples file.
    
    Verifies that the load_examples function correctly parses a properly formatted
    examples file with various line formats.
    """
    content = """
# This is a comment
Input Text 1 ::: a
Input Text 2 with spaces   :::    b
Input Text 3 ::: c

Another Input ::: d
# Another comment
Last one ::: x
    """
    examples_file = tmp_path / "examples.txt"
    examples_file.write_text(content, encoding='utf-8')

    expected_output = [
        {'text': 'Input Text 1', 'option': 'a'},
        {'text': 'Input Text 2 with spaces', 'option': 'b'},
        {'text': 'Input Text 3', 'option': 'c'},
        {'text': 'Another Input', 'option': 'd'},
        {'text': 'Last one', 'option': 'x'},
    ]

    actual_output = mcq_parser.load_examples(str(examples_file))
    assert actual_output == expected_output

def test_load_examples_empty_file(tmp_path):
    """Test loading an empty examples file, which should return an empty list."""
    examples_file = tmp_path / "empty_examples.txt"
    examples_file.write_text("", encoding='utf-8')
    assert mcq_parser.load_examples(str(examples_file)) == []

def test_load_examples_comments_only_file(tmp_path):
    """Test loading a file with only comments and blank lines."""
    content = """
# Comment 1
# Comment 2

# Comment 3
    """
    examples_file = tmp_path / "comments_only.txt"
    examples_file.write_text(content, encoding='utf-8')
    assert mcq_parser.load_examples(str(examples_file)) == []

def test_load_examples_invalid_lines(tmp_path, capsys):
    """
    Test loading a file with some invalid lines.
    
    Verifies that the load_examples function correctly handles lines
    that don't follow the expected format, skipping invalid ones and
    logging appropriate warnings.
    """
    content = """Valid Line 1 ::: a
Invalid Line 2 (no separator)
Valid Line 3 ::: c
Invalid Line 4 ::: missing option
Invalid Line 5 missing text ::: d"""
    examples_file = tmp_path / "invalid_lines.txt"
    examples_file.write_text(content, encoding='utf-8')

    expected_output = [
        {'text': 'Valid Line 1', 'option': 'a'},
        {'text': 'Valid Line 3', 'option': 'c'},
        {'text': 'Invalid Line 4', 'option': 'missing option'},
        {'text': 'Invalid Line 5 missing text', 'option': 'd'},
    ]

    actual_output = mcq_parser.load_examples(str(examples_file))
    assert actual_output == expected_output

    # Check warnings for the skipped line
    captured = capsys.readouterr()
    assert "Skipping invalid line 2" in captured.err or "Skipping invalid line 2" in captured.out
    assert "missing ':::'" in captured.err or "missing ':::'" in captured.out

    # Ensure warnings for lines that were *not* skipped are absent
    assert "Skipping invalid line 4" not in captured.err and "Skipping invalid line 4" not in captured.out
    assert "Skipping invalid line 5" not in captured.err and "Skipping invalid line 5" not in captured.out

def test_load_examples_file_not_found(tmp_path, capsys):
    """Test behavior when the examples file does not exist."""
    non_existent_file = tmp_path / "non_existent.txt"
    assert mcq_parser.load_examples(str(non_existent_file)) is None
    captured = capsys.readouterr()
    assert "Warning: Examples file not found" in captured.err or "Warning: Examples file not found" in captured.out

def test_load_examples_path_is_none():
    """Test behavior when None is passed as the path."""
    assert mcq_parser.load_examples(None) is None


# ============================================================================
# Tests for calculate_metric
# ============================================================================

# Sample data for calculate_metric tests
@pytest.fixture
def sample_metric_data():
    """Fixture providing sample data for metric calculation tests."""
    return pd.DataFrame({
        'language': ['en', 'en', 'en', 'fr', 'fr', 'en', 'fr', 'es', 'es', 'es'],
        'prompt_no': ['1', '1', '2', '1', '1', '1', '1', '3', '3', '3'],
        'final_prediction': ['a', 'b', 'c', 'a', 'a', 'x', None, 'd', 'c', 'd'],
        'reference_answer': ['a', 'a', 'c', 'b', 'a', 'a', 'b', 'd', 'd', 'd']
    })

def test_calculate_metric_basic(sample_metric_data):
    """
    Test basic accuracy calculation and grouping.
    
    Verifies that the calculate_metric function correctly groups data
    by language and prompt, filters out invalid predictions, and calculates
    accuracy metrics for each group.
    """
    results_df = mcq_parser.calculate_metric(sample_metric_data.copy(), 'final_prediction', 'reference_answer')

    assert not results_df.empty
    assert len(results_df) == 4  # 4 groups (en-1, en-2, fr-1, es-3)

    # Check en-1
    en1 = results_df[(results_df['language'] == 'en') & (results_df['prompt_no'] == '1')].iloc[0]
    assert en1['accuracy'] == 50.00
    assert en1['correct'] == 1
    assert en1['count_in_calc'] == 2
    assert en1['total_original'] == 3

    # Check en-2
    en2 = results_df[(results_df['language'] == 'en') & (results_df['prompt_no'] == '2')].iloc[0]
    assert en2['accuracy'] == 100.00
    assert en2['correct'] == 1
    assert en2['count_in_calc'] == 1
    assert en2['total_original'] == 1

    # Check fr-1
    fr1 = results_df[(results_df['language'] == 'fr') & (results_df['prompt_no'] == '1')].iloc[0]
    assert fr1['accuracy'] == 50.00
    assert fr1['correct'] == 1
    assert fr1['count_in_calc'] == 2
    assert fr1['total_original'] == 3

    # Check es-3
    es3 = results_df[(results_df['language'] == 'es') & (results_df['prompt_no'] == '3')].iloc[0]
    assert es3['accuracy'] == pytest.approx(66.67, abs=0.01)
    assert es3['correct'] == 2
    assert es3['count_in_calc'] == 3
    assert es3['total_original'] == 3

def test_calculate_metric_all_filtered(capsys):
    """Test when all predictions are None or 'x' and should be filtered out."""
    df = pd.DataFrame({
        'language': ['en', 'en'],
        'prompt_no': ['1', '1'],
        'final_prediction': ['x', None],
        'reference_answer': ['a', 'b']
    })
    results_df = mcq_parser.calculate_metric(df, 'final_prediction', 'reference_answer')
    assert results_df.empty
    captured = capsys.readouterr()
    assert "Warning: No valid predictions available for comparison" in captured.out

def test_calculate_metric_empty_input():
    """Test with an empty input DataFrame."""
    df = pd.DataFrame(columns=['language', 'prompt_no', 'final_prediction', 'reference_answer'])
    results_df = mcq_parser.calculate_metric(df, 'final_prediction', 'reference_answer')
    assert results_df.empty


# ============================================================================
# Tests for check_ollama_availability
# ============================================================================

# Define mock JSON responses
MOCK_MODELS_WITH_LLAMA3 = {
    "models": [
        {"name": "mistral:latest", "modified_at": "...", "size": 123},
        {"name": "llama3:latest", "modified_at": "...", "size": 456},
        {"name": "llama3:8b", "modified_at": "...", "size": 789},
    ]
}

MOCK_MODELS_WITHOUT_LLAMA3 = {
    "models": [
        {"name": "mistral:latest", "modified_at": "...", "size": 123},
        {"name": "phi:latest", "modified_at": "...", "size": 987},
    ]
}

MOCK_MODELS_INVALID_STRUCTURE = {"data": []}
MOCK_MODELS_NOT_A_LIST = {"models": "this is not a list"}


@patch('requests.get')
def test_check_ollama_availability_model_present(mock_get):
    """Test when the requested model exists in the Ollama list."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_MODELS_WITH_LLAMA3
    mock_get.return_value = mock_response

    assert mcq_parser.check_ollama_availability("llama3:latest") is True
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)

@patch('requests.get')
def test_check_ollama_availability_model_present_different_tag(mock_get):
    """Test when a model with the same base name but different tag exists."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_MODELS_WITH_LLAMA3
    mock_get.return_value = mock_response

    assert mcq_parser.check_ollama_availability("llama3:8b") is True
    assert mcq_parser.check_ollama_availability("llama3:70b") is False


@patch('requests.get')
def test_check_ollama_availability_model_absent(mock_get):
    """Test when the requested model does not exist in the Ollama list."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_MODELS_WITHOUT_LLAMA3
    mock_get.return_value = mock_response

    assert mcq_parser.check_ollama_availability("llama3:latest") is False

@patch('requests.get')
def test_check_ollama_availability_connection_error(mock_get, capsys):
    """Test connection error during the API call."""
    mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

    assert mcq_parser.check_ollama_availability("llama3:latest") is False
    captured = capsys.readouterr()
    assert "Ollama API check failed: Failed to connect" in captured.out

@patch('requests.get')
def test_check_ollama_availability_timeout(mock_get, capsys):
    """Test timeout during the API call."""
    mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

    assert mcq_parser.check_ollama_availability("llama3:latest") is False
    captured = capsys.readouterr()
    assert "Ollama API check failed: Request timed out" in captured.out

@patch('requests.get')
def test_check_ollama_availability_request_exception(mock_get, capsys):
    """Test other request exceptions during the API call."""
    mock_get.side_effect = requests.exceptions.RequestException("Some other error")

    assert mcq_parser.check_ollama_availability("llama3:latest") is False
    captured = capsys.readouterr()
    assert "Ollama API check failed: Some other error" in captured.out

@patch('requests.get')
def test_check_ollama_availability_bad_status(mock_get, capsys):
    """Test non-200 HTTP status code response."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not Found")
    mock_get.return_value = mock_response

    assert mcq_parser.check_ollama_availability("llama3:latest") is False
    captured = capsys.readouterr()
    assert "Ollama API check failed: Not Found" in captured.out

@patch('requests.get')
def test_check_ollama_availability_invalid_json(mock_get, capsys):
    """Test response that is not valid JSON."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_get.return_value = mock_response

    assert mcq_parser.check_ollama_availability("llama3:latest") is False
    captured = capsys.readouterr()
    assert "Failed to decode JSON response" in captured.out

@patch('requests.get')
def test_check_ollama_availability_json_missing_key(mock_get, capsys):
    """Test JSON response missing the 'models' key."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_MODELS_INVALID_STRUCTURE
    mock_get.return_value = mock_response

    assert mcq_parser.check_ollama_availability("llama3:latest") is False
    captured = capsys.readouterr()
    assert "Ollama API returned unexpected data format" in captured.out

@patch('requests.get')
def test_check_ollama_availability_json_models_not_list(mock_get, capsys):
    """Test JSON response where 'models' is not a list."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_MODELS_NOT_A_LIST
    mock_get.return_value = mock_response

    assert mcq_parser.check_ollama_availability("llama3:latest") is False
    captured = capsys.readouterr()
    assert "Ollama API returned unexpected 'models' format" in captured.out


# ============================================================================
# Tests for call_ollama_api
# ============================================================================

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_success_basic(mock_post, mock_sleep):
    """
    Test a successful API call returning a simple response.
    
    Verifies that the call_ollama_api function correctly formats the request
    to the Ollama API and processes a basic successful response.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": " a "}
    mock_post.return_value = mock_response

    prompt = "Extract option: A B C D"
    model = "test-model"
    valid_options = "abcdx"
    max_tokens = 1024

    result = mcq_parser.call_ollama_api(prompt, model, valid_options, max_tokens, enable_cot=False)

    assert result == "a"  # Expect stripped output
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "http://localhost:11434/api/generate"
    sent_payload = call_kwargs['json']
    assert sent_payload['model'] == model
    assert sent_payload['prompt'] == prompt
    assert sent_payload['stream'] is False
    assert sent_payload['options']['num_ctx'] == max_tokens
    assert sent_payload['options']['num_predict'] == 10
    assert isinstance(sent_payload['options']['stop'], list)
    mock_sleep.assert_not_called()

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_success_cot(mock_post, mock_sleep):
    """
    Test a successful API call with Chain-of-Thought enabled.
    
    Verifies that the call_ollama_api function correctly handles CoT responses,
    extracting the final answer after the thinking process.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "<think>\nThinking step 1.\nThinking step 2.\n</think>\n The final answer is: B "}
    mock_post.return_value = mock_response

    prompt = "Extract option with CoT: A B C D"
    model = "test-model-cot"
    valid_options = "abcdx"
    max_tokens = 2048

    result = mcq_parser.call_ollama_api(prompt, model, valid_options, max_tokens, enable_cot=True)

    assert result == "The final answer is: B"  # Expect stripped output
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    sent_payload = call_kwargs['json']
    assert sent_payload['model'] == model
    assert sent_payload['prompt'] == prompt
    assert sent_payload['options']['num_predict'] == 150
    assert 'stop' not in sent_payload['options']
    mock_sleep.assert_not_called()

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_success_cot_no_think_tag(mock_post, mock_sleep):
    """
    Test CoT handling when the API response lacks the <think> tag.

    Verifies that the function still processes the response correctly,
    stripping whitespace, even without the expected CoT structure when
    enable_cot=True is specified.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "  C  "} # Response without <think>...</think>
    mock_post.return_value = mock_response

    prompt = "Extract option with CoT: A B C D"
    model = "test-model-cot"
    # Using default valid_options and max_tokens

    result = mcq_parser.call_ollama_api(prompt, model, enable_cot=True)

    assert result == "C" # Expect stripped output, handled correctly despite missing tags
    mock_post.assert_called_once()
    # Verify basic call structure aspects were set correctly for CoT
    call_args, call_kwargs = mock_post.call_args
    sent_payload = call_kwargs['json']
    assert sent_payload['model'] == model
    assert sent_payload['prompt'] == prompt
    assert sent_payload['options']['num_predict'] == 150 # CoT increases predict length
    assert 'stop' not in sent_payload['options']       # CoT removes stop tokens
    mock_sleep.assert_not_called()

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_retry_on_500_error(mock_post, mock_sleep):
    """
    Test retry logic on HTTP 500 error, eventually succeeding.
    
    Verifies that the call_ollama_api function correctly implements retry
    logic when HTTP 500 errors occur, and returns the response when
    a subsequent attempt succeeds.
    """
    mock_500_response = MagicMock()
    mock_500_response.status_code = 500
    mock_500_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")

    mock_success_response = MagicMock()
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {"response": "a"}

    mock_post.side_effect = [
        mock_500_response,
        mock_500_response,
        mock_success_response
    ]

    result = mcq_parser.call_ollama_api("prompt", "model")

    assert result == "a"
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2
    assert mock_500_response.raise_for_status.call_count == 2

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_retry_on_connection_error(mock_post, mock_sleep, capsys):
    """
    Test retry logic specifically for requests.exceptions.ConnectionError.

    Verifies that the function retries the specified number of times upon
    encountering connection errors during the API call and returns the
    successful response if one occurs within the allowed retries. Also checks
    that appropriate warning messages are printed during retries.
    """
    mock_success_response = MagicMock()
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {"response": " d "} # Response with spaces

    # Simulate two connection errors followed by success
    mock_post.side_effect = [
        requests.exceptions.ConnectionError("Attempt 1 failed: Network unreachable"),
        requests.exceptions.ConnectionError("Attempt 2 failed: Could not resolve host"),
        mock_success_response
    ]

    prompt = "test prompt for connection error"
    model = "retry-test-model"
    result = mcq_parser.call_ollama_api(prompt, model) # Use defaults for other args

    assert result == "d" # Check successful result after retries, with stripping
    assert mock_post.call_count == 3 # Retried twice, succeeded on the 3rd attempt
    assert mock_sleep.call_count == 2 # Slept between the failed attempts

    # Check the arguments of the *last* (successful) call
    final_call_args, final_call_kwargs = mock_post.call_args
    assert final_call_args[0] == "http://localhost:11434/api/generate"
    sent_payload = final_call_kwargs['json']
    assert sent_payload['model'] == model
    assert sent_payload['prompt'] == prompt

    # Check that warnings were printed for the failed attempts
    captured = capsys.readouterr()
    assert "Retrying" in captured.out
    assert "Attempt 1 failed: Network unreachable" in captured.out
    assert "Attempt 2 failed: Could not resolve host" in captured.out

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_max_retries_fail(mock_post, mock_sleep, capsys):
    """
    Test when all retry attempts fail.
    
    Verifies that the call_ollama_api function correctly handles the case
    when all retry attempts fail, returning an empty string and logging
    appropriate error messages.
    """
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

    result = mcq_parser.call_ollama_api("prompt", "model")

    assert result == ""
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2
    captured = capsys.readouterr()
    # Check for either message indicating final failure
    assert "Max retries reached." in captured.out or "All API attempts failed." in captured.out

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_missing_response_key(mock_post, mock_sleep, capsys):
    """
    Test response JSON missing the 'response' key.
    
    Verifies that the call_ollama_api function correctly handles the case
    when the response JSON is missing the expected 'response' key.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"other_key": "some value"}
    mock_post.return_value = mock_response

    result = mcq_parser.call_ollama_api("prompt", "model")

    assert result == ""
    assert mock_post.call_count == 1 # No retries for missing key with .get default
    # Warnings related to retries should not be present
    captured = capsys.readouterr()
    assert "Ollama API returned invalid response" not in captured.out
    assert "All API attempts failed." not in captured.out

@patch('time.sleep', return_value=None)
@patch('requests.post')
def test_call_ollama_api_none_response_value(mock_post, mock_sleep, capsys):
    """
    Test response JSON where 'response' key has a None value.
    
    Verifies that the call_ollama_api function correctly handles the case
    when the 'response' key in the JSON response has a None value.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": None}
    mock_post.return_value = mock_response

    result = mcq_parser.call_ollama_api("prompt", "model")
    assert result == "" # Treat None response as empty string
    assert mock_post.call_count == 1 # No retries expected
    # Warnings related to retries should not be present
    captured = capsys.readouterr()
    assert "Ollama API returned invalid response" not in captured.out
    assert "All API attempts failed." not in captured.out


# ============================================================================
# Tests for extract_predicted_option_llm
# ============================================================================

# Sample examples for testing
SAMPLE_EXAMPLES = [
    {'text': 'Example input 1', 'option': 'a'},
    {'text': 'Example input 2', 'option': 'b'},
]

# Use patch for the function *called* by the one under test
@patch('mcq_parser.call_ollama_api')
def test_extract_llm_basic_success(mock_call_api):
    """
    Test successful extraction via LLM (mocked).
    
    Verifies that the extract_predicted_option_llm function correctly
    processes a successful LLM response.
    """
    mock_call_api.return_value = " c " # Simulate raw LLM response
    text = "Some long text about choices... the best is C."
    model = "test-llm"

    result = mcq_parser.extract_predicted_option_llm(
        text, model=model, option_format="letter", option_range="a-d"
    )

    assert result == "c"
    mock_call_api.assert_called_once()
    call_args, call_kwargs = mock_call_api.call_args
    # Check that the prompt contains the input text
    assert text in call_args[0]
    # Check other relevant args passed to call_ollama_api
    assert call_kwargs['model'] == model
    assert call_kwargs['valid_options'] == "abcdx" # Default includes 'x'
    assert call_kwargs['enable_cot'] is False


@patch('mcq_parser.call_ollama_api')
def test_extract_llm_with_examples_and_cot(mock_call_api):
    """
    Test LLM extraction with examples and Chain-of-Thought enabled.
    
    Verifies that the extract_predicted_option_llm function correctly formats
    the prompt with few-shot examples and CoT instructions when these options
    are enabled.
    """
    mock_call_api.return_value = " The final option is: a"
    text = "Input requiring examples and reasoning."
    model = "test-llm-advanced"

    result = mcq_parser.extract_predicted_option_llm(
        text, model=model, option_format="letter", option_range="a-d",
        enable_cot=True, examples=SAMPLE_EXAMPLES
    )

    assert result == "a"
    mock_call_api.assert_called_once()
    call_args, call_kwargs = mock_call_api.call_args
    prompt_sent = call_args[0]
    # Check prompt contains elements from examples and input text
    assert SAMPLE_EXAMPLES[0]['text'] in prompt_sent
    assert SAMPLE_EXAMPLES[1]['text'] in prompt_sent
    assert f"Input: {SAMPLE_EXAMPLES[0]['text']}" in prompt_sent
    assert f"Output: {SAMPLE_EXAMPLES[0]['option']}" in prompt_sent
    assert "--- EXAMPLES START ---" in prompt_sent
    assert "--- EXAMPLES END ---" in prompt_sent
    assert text in prompt_sent
    assert "<think>" in prompt_sent # CoT tag included
    # Check other relevant args
    assert call_kwargs['model'] == model
    assert call_kwargs['enable_cot'] is True


@patch('mcq_parser.call_ollama_api')
def test_extract_llm_no_answer_token(mock_call_api):
    """
    Test when the LLM should return the no_answer_token.
    
    Verifies that the extract_predicted_option_llm function correctly handles
    cases where the LLM returns the no_answer_token instead of a standard option.
    """
    mock_call_api.return_value = " x " # LLM indicates no answer
    text = "Cannot determine the answer."
    no_answer = "x"

    result = mcq_parser.extract_predicted_option_llm(
        text, no_answer_token=no_answer, option_format="letter", option_range="a-d"
    )

    assert result == no_answer
    mock_call_api.assert_called_once()
    call_args, call_kwargs = mock_call_api.call_args
    assert call_kwargs['valid_options'] == f"abcd{no_answer}"


@patch('mcq_parser.call_ollama_api')
def test_extract_llm_api_returns_empty(mock_call_api):
    """
    Test when the mocked call_ollama_api returns an empty string.
    
    Verifies that the extract_predicted_option_llm function correctly handles
    the case when the underlying API call returns an empty response.
    """
    mock_call_api.return_value = ""
    text = "Some input text."

    result = mcq_parser.extract_predicted_option_llm(text)

    assert result is None # Should return None if API fails or returns nothing valid
    mock_call_api.assert_called_once()


@patch('mcq_parser.call_ollama_api')
def test_extract_llm_api_returns_invalid(mock_call_api, capsys):
    """
    Test when the mocked call_ollama_api returns text not containing a valid option.
    
    Verifies that the extract_predicted_option_llm function correctly handles
    the case when the LLM returns text that doesn't contain a valid option,
    attempting to use a fallback regex extraction and ultimately returning None.
    """
    mock_call_api.return_value = "  invalid response "
    text = "Some input text."

    result = mcq_parser.extract_predicted_option_llm(
        text, option_format="letter", option_range="a-d"
    )

    assert result is None # Fallback regex within extract_llm also fails
    mock_call_api.assert_called_once()
    captured = capsys.readouterr()
    # Check for the specific warning from extract_predicted_option_llm
    assert "LLM extraction did not yield a valid single option" in captured.out


@patch('mcq_parser.call_ollama_api')
def test_extract_llm_api_returns_verbose_but_valid(mock_call_api):
    """
    Test when the LLM returns extra text but contains a valid option.
    
    Verifies that the extract_predicted_option_llm function correctly handles
    the case when the LLM returns verbose output that contains a valid option,
    using the fallback regex extraction to find and return the correct option.
    """
    mock_call_api.return_value = " The answer is clearly B, because reasons. "

    result = mcq_parser.extract_predicted_option_llm(
        "Input text", option_format="letter", option_range="a-d"
    )

    # The fallback regex search within extract_predicted_option_llm should find 'b'
    assert result == "b"
    mock_call_api.assert_called_once()


# Note: Testing truncation requires careful calculation or smarter mocks.
# This is a simplified check focusing on whether truncation marker appears.
@patch('mcq_parser.call_ollama_api')
def test_extract_llm_truncation(mock_call_api, capsys):
    """
    Test if text truncation occurs when text + examples exceed max_tokens.
    
    Verifies that the extract_predicted_option_llm function correctly implements
    truncation logic when the input text would exceed the available token limit,
    adding a truncation marker and logging appropriate warnings.
    """
    mock_call_api.return_value = " a "
    # Create very long text (estimation: > (500 - 200 - examples) * 4 chars)
    long_text = "word " * 500
    max_tokens = 500 # Relatively small max_tokens to force truncation

    result = mcq_parser.extract_predicted_option_llm(
        long_text, max_tokens=max_tokens, examples=SAMPLE_EXAMPLES
    )

    assert result == "a"
    mock_call_api.assert_called_once()
    call_args, call_kwargs = mock_call_api.call_args
    prompt_sent = call_args[0]
    assert "... [truncated]" in prompt_sent # Check if marker is present
    assert long_text not in prompt_sent # Original full text should NOT be in prompt
    # Check that print warning occurred
    captured = capsys.readouterr()
    assert "Text too long" in captured.out
    assert "truncating" in captured.out

# ============================================================================
# Tests for process_predictions
# ============================================================================

# Sample DataFrame fixture for processing tests
@pytest.fixture
def sample_processing_data():
    return pd.DataFrame({
        'language': ['en', 'en', 'fr', 'fr', 'es', 'es'],
        'prompt_no': ['1', '1', '2', '2', '3', '3'],
        'raw_prediction': [ # Original target column content
            'Answer: a',                 # Regex should find 'a'
            'Some text, maybe B?',       # Regex should find 'b'
            'Output is clearly C.',      # Regex should find 'c'
            'I cannot determine choice.',# Regex fails -> NA
            'Final Answer: x',           # Regex fails -> NA
            'Result: 5 (out of range)',  # Regex fails -> NA
        ],
        'reference': ['a', 'b', 'c', 'd', 'a', 'd'] # Reference answers
    })


# Test scenario 1: Regex only
def test_process_predictions_regex_only(sample_processing_data):
    """
    Test processing with only regex extraction.
    
    Verifies that the process_predictions function correctly extracts
    options using regex when LLM extraction is disabled.
    """
    df = sample_processing_data.copy()
    target_col = 'raw_prediction'
    reference_col = 'reference'

    processed_df = mcq_parser.process_predictions(
        df, target_col, reference_col, use_llm=False, skip_regex=False,
        option_format='letter', option_range='a-d', handle_x='random'
    )

    # --- Assertions for extracted_option ---
    extracted = processed_df['extracted_option']
    assert extracted.iloc[0] == 'a'
    assert extracted.iloc[1] == 'b'
    assert extracted.iloc[2] == 'c'
    assert pd.isna(extracted.iloc[3]) # Regex fails
    assert pd.isna(extracted.iloc[4]) # Regex fails for 'Final Answer: x'
    assert pd.isna(extracted.iloc[5]) # Regex fails

    # --- Assertions for verbalized_option ---
    verbalized = processed_df['verbalized_option']
    assert verbalized.iloc[0] == 'a'
    assert verbalized.iloc[1] == 'b'
    assert verbalized.iloc[2] == 'c'
    assert pd.isna(verbalized.iloc[3]) # NA propagates
    assert pd.isna(verbalized.iloc[4]) # NA propagates ('x' wasn't extracted)
    assert pd.isna(verbalized.iloc[5]) # NA propagates

    # --- Assertions for final labels (handle_x='random') ---
    final_labels = processed_df[target_col].tolist()
    assert final_labels[0] == 'a'
    assert final_labels[1] == 'b'
    assert final_labels[2] == 'c'
    # Random incorrect assignment where verbalized was NA
    assert final_labels[3] in {'a', 'b', 'c'} # ref=d
    assert final_labels[4] in {'b', 'c', 'd'} # ref=a
    assert final_labels[5] in {'a', 'b', 'c'} # ref=d

# Test scenario 2: Regex + LLM Fallback
@patch('mcq_parser.check_ollama_availability', return_value=True)
@patch('mcq_parser.extract_predicted_option_llm')
def test_process_predictions_regex_llm_fallback(mock_extract_llm, mock_check_ollama, sample_processing_data):
    """
    Test processing with regex + LLM fallback.
    
    Verifies that the process_predictions function correctly uses LLM extraction
    as a fallback mechanism when regex extraction fails for certain inputs.
    """
    df = sample_processing_data.copy()
    target_col = 'raw_prediction'
    reference_col = 'reference'

    # Configure mock LLM for rows where regex failed (indices 3, 4, 5)
    mock_extract_llm.side_effect = lambda text, **kwargs: {
        'I cannot determine choice.': 'd',
        'Final Answer: x': 'x',           # LLM correctly extracts 'x'
        'Result: 5 (out of range)': 'x'  # LLM returns 'x' for out-of-range
    }.get(text) # Default to None if text doesn't match

    processed_df = mcq_parser.process_predictions(
        df, target_col, reference_col, use_llm=True, skip_regex=False,
        option_format='letter', option_range='a-d', handle_x='keep' # Use handle_x='keep'
    )

    # --- Assertions for extracted_option ---
    extracted = processed_df['extracted_option']
    assert extracted.iloc[0] == 'a' # Regex
    assert extracted.iloc[1] == 'b' # Regex
    assert extracted.iloc[2] == 'c' # Regex
    assert extracted.iloc[3] == 'd' # LLM
    assert extracted.iloc[4] == 'x' # LLM
    assert extracted.iloc[5] == 'x' # LLM

    # --- Assertions for verbalized_option ---
    verbalized = processed_df['verbalized_option']
    assert verbalized.iloc[0] == 'a'
    assert verbalized.iloc[1] == 'b'
    assert verbalized.iloc[2] == 'c'
    assert verbalized.iloc[3] == 'd'
    assert verbalized.iloc[4] == 'x' # Verbalizer passes 'x'
    assert verbalized.iloc[5] == 'x' # Verbalizer passes 'x'

    # --- Assertions for final labels (handle_x='keep') ---
    expected_final = ['a', 'b', 'c', 'd', 'x', 'x']
    assert processed_df[target_col].tolist() == expected_final

    # Verify mocks
    mock_check_ollama.assert_called_once()
    # LLM called only for indices 3, 4, 5 where regex initially failed
    assert mock_extract_llm.call_count == 3
    call_texts = {call.args[0] for call in mock_extract_llm.call_args_list}
    assert 'I cannot determine choice.' in call_texts
    assert 'Final Answer: x' in call_texts
    assert 'Result: 5 (out of range)' in call_texts

# Test scenario 3: LLM Only (Skip Regex) - VERIFYING THE FIX
@patch('mcq_parser.check_ollama_availability', return_value=True)
@patch('mcq_parser.extract_predicted_option') # Mock the REGEX function
@patch('mcq_parser.extract_predicted_option_llm') # Mock the LLM function
def test_process_predictions_llm_only_skip_regex_works(mock_extract_llm, mock_extract_regex, mock_check_ollama, sample_processing_data):
    """
    Test processing with LLM only, ensuring regex is actually skipped.
    
    This test verifies that when skip_regex=True, the extraction pipeline
    bypasses the regex extraction step completely and relies solely on the LLM.
    """
    df = sample_processing_data.copy()
    target_col = 'raw_prediction'
    reference_col = 'reference'

    # Configure mock LLM for ALL rows (as regex is skipped)
    llm_responses = {
        'Answer: a': 'a',
        'Some text, maybe B?': 'b',
        'Output is clearly C.': 'c',
        'I cannot determine choice.': 'd',
        'Final Answer: x': 'x',
        'Result: 5 (out of range)': None # LLM returns None for this one
    }
    # Use a function for side_effect to check input text
    def llm_side_effect(text, **kwargs):
        print(f"  Mock LLM received text: '{text[:50]}...'") # Debug print
        return llm_responses.get(text)

    mock_extract_llm.side_effect = llm_side_effect

    processed_df = mcq_parser.process_predictions(
        df, target_col, reference_col, use_llm=True, skip_regex=True, # skip_regex = True
        option_format='letter', option_range='a-d', handle_x='keep' # Use handle_x='keep' for easier checking
    )

    # --- Assertions ---
    # 1. Assert Regex was NOT called
    mock_extract_regex.assert_not_called()

    # 2. Assert Ollama Check WAS called
    mock_check_ollama.assert_called_once()

    # 3. Assert LLM WAS called for ALL rows
    assert mock_extract_llm.call_count == len(df)

    # 4. Check extracted_option column contains LLM results
    extracted = processed_df['extracted_option'].tolist() # Use list for direct comparison
    expected_extracted = ['a', 'b', 'c', 'd', 'x', pd.NA] # Expected results from mock LLM
     # Compare element by element, handling NA
    for i, (actual, expected) in enumerate(zip(extracted, expected_extracted)):
        if pd.isna(expected):
            assert pd.isna(actual), f"Mismatch at index {i}: Expected NA, got {actual}"
        else:
            assert actual == expected, f"Mismatch at index {i}: Expected {expected}, got {actual}"


    # 5. Check verbalized_option column
    verbalized = processed_df['verbalized_option'].tolist()
    expected_verbalized = ['a', 'b', 'c', 'd', 'x', pd.NA] # Verbalizer passes valid options/x, keeps NA
    for i, (actual, expected) in enumerate(zip(verbalized, expected_verbalized)):
        if pd.isna(expected):
            assert pd.isna(actual), f"Mismatch at index {i} (verbalized): Expected NA, got {actual}"
        else:
            assert actual == expected, f"Mismatch at index {i} (verbalized): Expected {expected}, got {actual}"

    # 6. Check final labels (handle_x='keep')
    final_labels = processed_df[target_col].tolist()
    # assign_label keeps 'a','b','c','d','x'. NA becomes random incorrect.
    expected_final = ['a', 'b', 'c', 'd', 'x', None] # Placeholder for random
    assert final_labels[:5] == expected_final[:5]
    # For the last one (NA -> random incorrect, ref='d'), check it's valid random
    assert final_labels[5] in {'a', 'b', 'c'}


# Test scenario 4: LLM enabled but unavailable
@patch('mcq_parser.check_ollama_availability', return_value=False)
@patch('mcq_parser.extract_predicted_option_llm')
def test_process_predictions_llm_unavailable(mock_extract_llm, mock_check_ollama, sample_processing_data, capsys):
    """
    Test processing when use_llm=True but Ollama is unavailable.
    
    Verifies that the process_predictions function correctly handles the case
    when LLM extraction is enabled but the Ollama service is not available,
    falling back to regex-only extraction.
    """
    df = sample_processing_data.copy()
    target_col = 'raw_prediction'
    reference_col = 'reference'
    processed_df = mcq_parser.process_predictions(
        df, target_col, reference_col, use_llm=True, skip_regex=False,
        option_format='letter', option_range='a-d', handle_x='keep'
    )

    # --- Assertions for extracted_option (only regex results) ---
    extracted = processed_df['extracted_option']
    assert extracted.iloc[0] == 'a'
    assert extracted.iloc[1] == 'b'
    assert extracted.iloc[2] == 'c'
    assert pd.isna(extracted.iloc[3]) # Regex fails
    assert pd.isna(extracted.iloc[4]) # Regex fails
    assert pd.isna(extracted.iloc[5]) # Regex fails

    # --- Assertions for verbalized_option ---
    verbalized = processed_df['verbalized_option']
    assert verbalized.iloc[0] == 'a'
    assert verbalized.iloc[1] == 'b'
    assert verbalized.iloc[2] == 'c'
    assert pd.isna(verbalized.iloc[3]) # NA propagates
    assert pd.isna(verbalized.iloc[4]) # NA propagates
    assert pd.isna(verbalized.iloc[5]) # NA propagates

    # --- Assertions for final labels (handle_x='keep') ---
    final_labels = processed_df[target_col].tolist()
    assert final_labels[0] == 'a'
    assert final_labels[1] == 'b'
    assert final_labels[2] == 'c'
    # Random incorrect assignment where verbalized was NA
    assert final_labels[3] in {'a', 'b', 'c'} # ref=d
    assert final_labels[4] in {'b', 'c', 'd'} # ref=a
    assert final_labels[5] in {'a', 'b', 'c'} # ref=d

    # Verify mocks
    mock_check_ollama.assert_called_once()
    mock_extract_llm.assert_not_called()
    captured = capsys.readouterr()
    assert "Ollama API not available" in captured.out

# Test scenario 5: Skip Regex, LLM Unavailable
@patch('mcq_parser.check_ollama_availability', return_value=False)
@patch('mcq_parser.extract_predicted_option') # Mock regex
@patch('mcq_parser.extract_predicted_option_llm') # Mock llm
def test_process_predictions_skip_regex_llm_unavailable(mock_extract_llm, mock_extract_regex, mock_check_ollama, sample_processing_data, capsys):
    """
    Test skipping regex when LLM is also unavailable.
    
    Verifies that the process_predictions function correctly handles the
    edge case when regex extraction is disabled (skip_regex=True) but
    LLM extraction is also unavailable, resulting in no extraction being
    performed and random assignment being used for all predictions.
    """
    df = sample_processing_data.copy()
    target_col = 'raw_prediction'
    reference_col = 'reference'

    processed_df = mcq_parser.process_predictions(
        df, target_col, reference_col, use_llm=True, skip_regex=True, # Skip regex!
        option_format='letter', option_range='a-d', handle_x='keep'
    )

    # --- Assertions ---
    # 1. Assert Regex was NOT called
    mock_extract_regex.assert_not_called()

    # 2. Assert Ollama Check WAS called
    mock_check_ollama.assert_called_once()

    # 3. Assert LLM was NOT called (because unavailable)
    mock_extract_llm.assert_not_called()

    # 4. Assert extracted_option is all NA
    assert processed_df['extracted_option'].isna().all()

    # 5. Assert verbalized_option is all NA
    assert processed_df['verbalized_option'].isna().all()

    # 6. Check final labels (handle_x='keep') - all should be random incorrect
    final_labels = processed_df[target_col].tolist()
    refs = df[reference_col].tolist()
    possible_options = set("abcd")
    for i, label in enumerate(final_labels):
        incorrect_options = possible_options - {refs[i]} if refs[i] in possible_options else possible_options
        assert label in incorrect_options

    # 7. Check Warnings
    captured = capsys.readouterr()
    assert "Skipping regex extraction" in captured.out
    assert "Ollama API not available" in captured.out
    assert "Regex was skipped and LLM is unavailable" in captured.out