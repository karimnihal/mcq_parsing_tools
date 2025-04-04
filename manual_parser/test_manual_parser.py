import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock
import manual_parser
import shutil

# Sample test data for different file formats and configurations
INPUT_DATA_TSV_DEFAULT_INDEX = """\tlanguage\tprompt_no\tsource_col\tother_col
id1\thau\tp1\tSource text one.\tans1
id2\tibo\tp1\tSource text two.\tans2
id3\tsot\tp2\tSource text three.\tans3
id4\teng\tp2\tSource text four.\tans4
"""

INPUT_DATA_TSV_NAMED_INDEX = """uid\tlanguage\tprompt_no\tsource_col\tother_col
id1\thau\tp1\tSource text one.\tans1
id2\tibo\tp1\tSource text two.\tans2
id3\tsot\tp2\tSource text three.\tans3
id4\teng\tp2\tSource text four.\tans4
"""

INPUT_DATA_CSV_NAMED_INDEX = """uid,language,prompt_no,source_col,other_col
id1,hau,p1,Source text one.,ans1
id2,ibo,p1,Source text two.,ans2
id3,sot,p2,Source text three.,ans3
id4,eng,p2,Source text four.,ans4
"""

INPUT_DATA_TSV_NUMERIC_INDEX = """id_num\tlanguage\tprompt_no\tsource_col\tother_col
101\thau\tp1\tSource text one.\tans1
102\tibo\tp1\tSource text two.\tans2
103\tsot\tp2\tSource text three.\tans3
104\teng\tp2\tSource text four.\tans4
"""

OUTPUT_DATA_TSV_EXISTING = """\tlanguage\tprompt_no\tsource_col\tother_col\tparsed_col
id1\thau\tp1\tSource text one.\tans1\ta
id2\tibo\tp1\tSource text two.\tans2\t
id3\tsot\tp2\tSource text three.\tans3\tx
id4\teng\tp2\tSource text four.\tans4\t
"""

OUTPUT_DATA_TSV_EXISTING_NAMED = """uid\tlanguage\tprompt_no\tsource_col\tother_col\tparsed_col
id1\thau\tp1\tSource text one.\tans1\ta
id2\tibo\tp1\tSource text two.\tans2\t
id3\tsot\tp2\tSource text three.\tans3\tx
id4\teng\tp2\tSource text four.\tans4\t
"""

OUTPUT_DATA_CSV_EXISTING_NAMED = """uid,language,prompt_no,source_col,other_col,parsed_col
id1,hau,p1,Source text one.,ans1,a
id2,ibo,p1,Source text two.,ans2,
id3,sot,p2,Source text three.,ans3,x
id4,eng,p2,Source text four.,ans4,
"""

OUTPUT_DATA_MISSING_PARSED = """uid\tlanguage\tprompt_no\tsource_col\tother_col
id1\thau\tp1\tSource text one.\tans1
id2\tibo\tp1\tSource text two.\tans2
id3\tsot\tp2\tSource text three.\tans3
id4\teng\tp2\tSource text four.\tans4
"""

OUTPUT_DATA_TSV_ALL_VALID = """\tlanguage\tprompt_no\tsource_col\tother_col\tparsed_col
id1\thau\tp1\tSource text one.\tans1\ta
id2\tibo\tp1\tSource text two.\tans2\tb
id3\tsot\tp2\tSource text three.\tans3\tc
id4\teng\tp2\tSource text four.\tans4\td
"""

def run_main_with_args(args_list, monkeypatch, mock_inputs=None):
    """
    Helper function to run the script's main function with mocked arguments and inputs.
    
    Args:
        args_list: Command line arguments to pass to the script
        monkeypatch: Pytest monkeypatch fixture
        mock_inputs: List of inputs to simulate user interaction (defaults to ['q'])
    """
    monkeypatch.setattr(sys, 'argv', ['manual_parser.py'] + args_list)
    monkeypatch.setattr(manual_parser, 'clear_screen', lambda: None)

    processed_mock_inputs = list(mock_inputs) if mock_inputs is not None else ['q']
    mock_input_iter = iter(processed_mock_inputs)

    def mock_input_func(*args, **kwargs):
        prompt_text = args[0] if args else ''
        next_val = next(mock_input_iter, 'q')  # Default 'q' to avoid hangs
        print(f"Mock Input: Prompt='{prompt_text[:100]}...', Returning='{next_val}'")
        return next_val

    with patch('builtins.input', mock_input_func):
         manual_parser.main()


# Basic Creation Tests
def test_create_new_file_indexed_default(tmp_path, monkeypatch):
    """Test creating a new output file using default pandas index."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_DEFAULT_INDEX)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_output",
            "--index-col", "Unnamed: 0"]
    mock_inputs = ['y', 'a', 'b', 'c', 'q']  # Resume 'y', ans 1, 2, 3, quit 4

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert df_out.columns[0] == 'Unnamed: 0'
    assert list(df_out.columns) == ['Unnamed: 0', 'language', 'prompt_no', 'source_col', 'other_col', 'manual_output']
    assert len(df_out) == 4
    assert df_out.iloc[0]['manual_output'] == 'a'
    assert df_out.iloc[1]['manual_output'] == 'b'
    assert df_out.iloc[2]['manual_output'] == 'c'
    assert df_out.iloc[3]['manual_output'] == ''

def test_create_new_file_indexed_named(tmp_path, monkeypatch):
    """Test creating a new output file using a named index column."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_output",
            "--index-col", "uid"]
    mock_inputs = ['y', 'a', 'b', 'c', 'd']  # Resume 'y', ans 1, 2, 3, 4

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert list(df_out.columns) == ['uid', 'language', 'prompt_no', 'source_col', 'other_col', 'manual_output']
    assert len(df_out) == 4
    assert df_out.iloc[0]['manual_output'] == 'a'
    assert df_out.iloc[1]['manual_output'] == 'b'
    assert df_out.iloc[2]['manual_output'] == 'c'
    assert df_out.iloc[3]['manual_output'] == 'd'

def test_numeric_index(tmp_path, monkeypatch):
    """Test using a purely numeric index column."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NUMERIC_INDEX)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_output",
            "--index-col", "id_num"]  # Use numeric index name
    mock_inputs = ['y', 'a', 'b', 'c', 'd']

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert list(df_out.columns) == ['id_num', 'language', 'prompt_no', 'source_col', 'other_col', 'manual_output']
    assert len(df_out) == 4
    assert df_out.iloc[0]['id_num'] == '101'  # Verify index values are strings
    assert df_out.iloc[0]['manual_output'] == 'a'
    assert df_out.iloc[3]['manual_output'] == 'd'

# Update/Resume Tests
def test_update_existing_file_indexed_default(tmp_path, monkeypatch):
    """Test updating existing file with default pandas index."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_DEFAULT_INDEX)
    output_file.write_text(OUTPUT_DATA_TSV_EXISTING)  # has '', 'x', '' as invalid
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col",
            "--index-col", "Unnamed: 0"]
    mock_inputs = ['y', 'b', 's', 'd', 'q']  # Resume 'y' (start id2), ans 'b', skip id3 ('x'), ans 'd' for id4

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert df_out.columns[0] == 'Unnamed: 0'
    assert list(df_out.columns) == ['Unnamed: 0', 'language', 'prompt_no', 'source_col', 'other_col', 'parsed_col']
    assert len(df_out) == 4
    assert df_out.iloc[0]['parsed_col'] == 'a'  # Before resume
    assert df_out.iloc[1]['parsed_col'] == 'b'  # Changed '' to 'b'
    assert df_out.iloc[2]['parsed_col'] == 'x'  # Skipped, remained 'x'
    assert df_out.iloc[3]['parsed_col'] == 'd'  # Changed '' to 'd'

def test_update_existing_file_indexed_named(tmp_path, monkeypatch):
    """Test updating existing file with named index column."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)
    output_file.write_text(OUTPUT_DATA_TSV_EXISTING_NAMED)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col",
            "--index-col", "uid"]
    mock_inputs = ['y', 'b', 's', 'd']  # Resume 'y' (start id2), ans 'b', skip id3 ('x'), ans 'd' for id4

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert list(df_out.columns) == ['uid', 'language', 'prompt_no', 'source_col', 'other_col', 'parsed_col']
    assert len(df_out) == 4
    assert df_out.iloc[0]['parsed_col'] == 'a'
    assert df_out.iloc[1]['parsed_col'] == 'b'
    assert df_out.iloc[2]['parsed_col'] == 'x'
    assert df_out.iloc[3]['parsed_col'] == 'd'

def test_update_existing_file_indexed_default_resume_no(tmp_path, monkeypatch):
    """Test choosing 'n' for resume - should start from row 1."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_DEFAULT_INDEX)
    output_file.write_text(OUTPUT_DATA_TSV_EXISTING)  # existing has 'a', '', 'x', ''
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col",
            "--index-col", "Unnamed: 0"]
    # Resume 'n', ans 'd' for row 1, 'b' for row 2, skip row 3, ans 'c' for row 4
    mock_inputs = ['n', 'd', 'b', 's', 'c']

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert df_out.columns[0] == 'Unnamed: 0'
    assert len(df_out) == 4
    assert df_out.iloc[0]['parsed_col'] == 'd'  # Changed 'a' to 'd'
    assert df_out.iloc[1]['parsed_col'] == 'b'  # Changed '' to 'b'
    assert df_out.iloc[2]['parsed_col'] == 'x'  # Skipped 'x'
    assert df_out.iloc[3]['parsed_col'] == 'c'  # Changed '' to 'c'

def test_update_existing_overwrite(tmp_path, monkeypatch):
    """Test overwriting an existing valid answer."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)
    # Output has 'a', '', 'x', '' initially
    output_file.write_text(OUTPUT_DATA_TSV_EXISTING_NAMED)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col",
            "--index-col", "uid"]
    # Resume 'n', overwrite 'a' with 'd', fill '', skip 'x', fill ''
    mock_inputs = ['n', 'd', 'b', 's', 'c']

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert len(df_out) == 4
    assert df_out.iloc[0]['parsed_col'] == 'd'  # Overwritten
    assert df_out.iloc[1]['parsed_col'] == 'b'
    assert df_out.iloc[2]['parsed_col'] == 'x'
    assert df_out.iloc[3]['parsed_col'] == 'c'

def test_update_all_valid_resume(tmp_path, monkeypatch):
    """Test resuming when all answers are already valid."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_DEFAULT_INDEX)
    # Write the expected initial/final state to the output file
    output_file.write_text(OUTPUT_DATA_TSV_ALL_VALID)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col",
            "--index-col", "Unnamed: 0"]
    # Simulate 'y' resume -> All valid, loops from start -> Quit immediately
    mock_inputs = ['y', 'q']  # Choose resume 'y', then quit immediately

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert df_out.columns[0] == 'Unnamed: 0'
    assert len(df_out) == 4
    
    # Read the ORIGINAL expected data string again and compare DataFrames
    from io import StringIO  # Used to read string as file
    df_expected = pd.read_csv(StringIO(OUTPUT_DATA_TSV_ALL_VALID), sep='\t', dtype=str, keep_default_na=False).fillna('')
    # Use pandas testing function for robust comparison
    pd.testing.assert_frame_equal(df_out, df_expected)

# Output File State Tests
def test_update_output_missing_parsed_col(tmp_path, monkeypatch):
    """Test updating when output file exists but is missing the parsed column."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)
    # Output file exists but doesn't have 'manual_label' column
    output_file.write_text(OUTPUT_DATA_MISSING_PARSED)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_label",  # Request new col name
            "--index-col", "uid"]
    # Resume 'y', answer all rows
    mock_inputs = ['y', 'a', 'b', 'c', 'd']

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert 'manual_label' in df_out.columns  # Check column was added
    assert len(df_out) == 4
    assert df_out.iloc[0]['manual_label'] == 'a'
    assert df_out.iloc[3]['manual_label'] == 'd'

# Format/Options Tests
def test_update_existing_csv_indexed_named(tmp_path, monkeypatch):
    """Test updating an existing CSV file using a named index."""
    input_file = tmp_path / "input.csv"; output_file = tmp_path / "output.csv"
    input_file.write_text(INPUT_DATA_CSV_NAMED_INDEX)
    output_file.write_text(OUTPUT_DATA_CSV_EXISTING_NAMED)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col",
            "--index-col", "uid",
            "--delimiter", ","]  # Specify CSV delimiter
    mock_inputs = ['y', 'b', 's', 'd']  # Resume 'y', ans 'b', skip 'x', ans 'd'

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep=',', dtype=str, keep_default_na=False).fillna('')  # Read as CSV
    assert list(df_out.columns) == ['uid', 'language', 'prompt_no', 'source_col', 'other_col', 'parsed_col']
    assert len(df_out) == 4
    assert df_out.iloc[0]['parsed_col'] == 'a'
    assert df_out.iloc[1]['parsed_col'] == 'b'
    assert df_out.iloc[2]['parsed_col'] == 'x'
    assert df_out.iloc[3]['parsed_col'] == 'd'


def test_options_a_f(tmp_path, monkeypatch):
    """Test using options a-f."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)  # Use 4 rows of standard input
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_f",
            "--index-col", "uid",
            "--option-format", "letter", "--option-range", "a-f"]
    mock_inputs = ['y', 'a', 'f', 'e', 'x']  # Answer using 'f', 'e', 'x'

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert len(df_out) == 4
    assert df_out.iloc[0]['parsed_f'] == 'a'
    assert df_out.iloc[1]['parsed_f'] == 'f'
    assert df_out.iloc[2]['parsed_f'] == 'e'
    assert df_out.iloc[3]['parsed_f'] == 'x'


def test_options_1_6_custom_token(tmp_path, monkeypatch):
    """Test using options 1-6 and a custom no-answer token."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)  # Use 4 rows of standard input
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_num",
            "--index-col", "uid",
            "--option-format", "number", "--option-range", "1-6",
            "--no-answer-token", "na"]  # Custom token
    mock_inputs = ['y', '1', '6', '5', 'na']  # Answer using numbers and 'na'

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert len(df_out) == 4
    assert df_out.iloc[0]['parsed_num'] == '1'
    assert df_out.iloc[1]['parsed_num'] == '6'
    assert df_out.iloc[2]['parsed_num'] == '5'
    assert df_out.iloc[3]['parsed_num'] == 'na'

# No Index Mode Tests
def test_no_index_mode_create(tmp_path, monkeypatch):
    """Test creating output in no-index mode."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_data_no_index = """language\tprompt_no\tsource_col\tother_col
hau\tp1\tSource text one.\tans1
ibo\tp1\tSource text two.\tans2
sot\tp2\tSource text three.\tans3
"""
    input_file.write_text(input_data_no_index)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_label",
            "--no-index"]
    mock_inputs = ['y', 'x', 'a', 'b']

    run_main_with_args(args, monkeypatch, mock_inputs)

    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert list(df_out.columns) == ['language', 'prompt_no', 'source_col', 'other_col', 'manual_label']
    assert len(df_out) == 3
    assert df_out.iloc[0]['manual_label'] == 'x'
    assert df_out.iloc[1]['manual_label'] == 'a'
    assert df_out.iloc[2]['manual_label'] == 'b'

def test_no_index_mode_update_mismatch(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_data_no_index = """language\tsource_col
hau\tSource text one.
ibo\tSource text two.
sot\tSource text three.
""" # 3 rows input
    output_data_existing = """language\tsource_col\tparsed_col
hau\tSource text one.\ta
ibo\tSource text two.\tb
""" # 2 rows output
    input_file.write_text(input_data_no_index)
    output_file.write_text(output_data_existing)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col",
            "--no-index"]
    mock_inputs = ['y', 'c', 'd', 'x']

    run_main_with_args(args, monkeypatch, mock_inputs)

    captured = capsys.readouterr()
    assert "Warning: Row count mismatch" in captured.out
    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert len(df_out) == 3
    assert list(df_out.columns) == ['language', 'source_col', 'parsed_col']
    assert df_out.iloc[0]['parsed_col'] == 'c'
    assert df_out.iloc[1]['parsed_col'] == 'd'
    assert df_out.iloc[2]['parsed_col'] == 'x'

# Error Handling Tests
def test_missing_input_file(tmp_path, monkeypatch):
    """Test handling of non-existent input file."""
    args = ["--input", str(tmp_path / "nonexistent.tsv"), "--output", str(tmp_path / "output.tsv"),
            "--source-col", "colA", "--parsed-col", "colB", "--index-col", "id"]
    with pytest.raises(SystemExit): run_main_with_args(args, monkeypatch)

def test_missing_source_column(tmp_path, monkeypatch):
    """Test handling of missing source column in input file."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_data = "uid\tlanguage\tother\nid1\thau\tval1"
    input_file.write_text(input_data)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_output", "--index-col", "uid"]
    with pytest.raises(SystemExit): run_main_with_args(args, monkeypatch)

def test_missing_index_column(tmp_path, monkeypatch):
    """Test handling of missing index column in input file."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_data = "language\tsource_col\tother\nhau\tSource text one.\tans1"
    input_file.write_text(input_data)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_output", "--index-col", "uid"]
    with pytest.raises(SystemExit): run_main_with_args(args, monkeypatch)

def test_missing_index_column_in_output(tmp_path, monkeypatch):
    """Test handling of missing index column in output file."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)
    output_data_missing_index = "language\tprompt_no\tsource_col\tparsed_col\nhau\tp1\tSource text one.\ta"
    output_file.write_text(output_data_missing_index)
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "parsed_col", "--index-col", "uid"]
    mock_inputs = ['y', 'q']
    with pytest.raises(SystemExit): run_main_with_args(args, monkeypatch, mock_inputs)

def test_arg_validation_no_index_method(monkeypatch):
    """Test validation of either --index-col or --no-index is required."""
    args_list = ['manual_parser.py', "--input", "in.tsv", "--output", "out.tsv",
                 "--source-col", "s", "--parsed-col", "p"]
    monkeypatch.setattr(sys, 'argv', args_list)
    with patch('argparse.ArgumentParser.error') as mock_error:
        mock_error.side_effect = SystemExit
        with pytest.raises(SystemExit): manual_parser.parse_arguments()
        mock_error.assert_called_once()

@pytest.mark.parametrize(
    "option_format, option_range, no_answer_token, expected_options",
    [ ("letter", "a-d", "x", ['a', 'b', 'c', 'd', 'x']),
      ("letter", "a-f", "x", ['a', 'b', 'c', 'd', 'e', 'f', 'x']),
      ("number", "1-4", "0", ['0', '1', '2', '3', '4']),
      ("number", "1-6", "na", ['1', '2', '3', '4', '5', '6', 'na']),
      ("letter", "a-d", "", ['a', 'b', 'c', 'd']), ]
)
def test_generate_valid_options(option_format, option_range, no_answer_token, expected_options):
    """Test generation of valid option sets based on format, range and token."""
    result = manual_parser.generate_valid_options(option_format, option_range, no_answer_token)
    assert sorted(result) == sorted(expected_options)

def test_empty_output_file(tmp_path, monkeypatch):
    """Test handling of empty output file."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.write_text(INPUT_DATA_TSV_NAMED_INDEX)
    output_file.touch()
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_output", "--index-col", "uid"]
    mock_inputs = ['y', 'a', 'b', 'c', 'd']
    run_main_with_args(args, monkeypatch, mock_inputs)
    assert output_file.exists()
    df_out = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False).fillna('')
    assert len(df_out) == 4
    assert df_out.iloc[0]['manual_output'] == 'a'
    assert df_out.iloc[1]['manual_output'] == 'b'
    assert df_out.iloc[2]['manual_output'] == 'c'
    assert df_out.iloc[3]['manual_output'] == 'd'

def test_empty_input_file(tmp_path, monkeypatch):
    """Test handling of an empty input file."""
    input_file = tmp_path / "input.tsv"; output_file = tmp_path / "output.tsv"
    input_file.touch()
    args = ["--input", str(input_file), "--output", str(output_file),
            "--source-col", "source_col", "--parsed-col", "manual_output", "--index-col", "uid"]
    with pytest.raises(SystemExit): # Should exit because file is empty or columns won't be found
        run_main_with_args(args, monkeypatch)