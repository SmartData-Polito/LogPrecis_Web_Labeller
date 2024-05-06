"""File containing the basic functions shared among methods for data and text processing
"""

import json
import os
import re
import pandas as pd
import seaborn as sns


def read_json_file(filepath):
    """Simple function that:
        - receives as input a filepath
        - checks if that links to a previously saved json file
        - reads it.
    Args:
        filepath (str): Path toward 'json' file (e.g., configuration).
    Returns:
        dict: Retrieved parameters (e.g., path to data if configuration, etc.)
    """
    assert (
        os.path.isfile(filepath) and os.path.splitext(filepath)[-1] == ".json"
    ), "Error: Expecting an existing json file!"
    with open(file=filepath, mode="r", encoding="utf-8") as f:
        retrieved_parameters = json.load(fp=f)
    return retrieved_parameters


def load_data(path, input_file):
    """Function that loads the input data using a Pandas Dataframe.
    Args:
        path (str): Path toward data.
        input_file (str): Filename (name of the dataset).

    Returns:
        DataFrame: pandas DataFrame containing the data.
    """
    file_format = os.path.splitext(input_file)[-1]
    filepath = os.path.join(path, input_file)
    assert file_format in [
        ".csv",
        ".parquet",
        ".json",
    ], "Error: Expecting an existing 'csv', 'parquet' or 'json' file!"
    if file_format == ".csv":
        df = pd.read_csv(filepath)
    elif file_format == ".parquet":
        df = pd.read_parquet(filepath)
    elif file_format == ".json":
        df = pd.read_json(filepath)
    return df


def save_data(df, output_path, file):
    """Save DataFrame to a Parquet file.
    Args:
        df (DataFrame): The DataFrame to be saved.
        output_path (str): Output path
        file (str): The name of the Parquet file.
    """
    # 1. Extract filename without extension
    filename = extract_file_formatless(file)
    # 2. Make sure that folder exists
    os.makedirs(output_path, exist_ok=True)
    # 3. Create output filepath
    file_path = os.path.join(output_path, f"{filename}.parquet")
    # 4. Save dataframe
    df.to_parquet(file_path, index=False)


def save_json(dictionary, output_path, file):
    """Function that saves a dictionary into a json file.
    Args:
        dictionary (dict): Dictionary we want to export
        output_path (str): Output path
        file (str): Name of the json file
    """
    # 1. Extract filename without extension
    filename = extract_file_formatless(file)
    # 2. Make sure that folder exists
    os.makedirs(output_path, exist_ok=True)
    # 3. Create output filepath
    file_path = os.path.join(output_path, f"{filename}.json")
    # 4. Save dataframe
    with open(file=file_path, mode="w+", encoding="utf-8") as f:
        json.dump(dictionary, fp=f)


def get_labels_2_colors(labels, seaborn_palette):
    """Function to generate a mapping colors to labels, given a list of labels and a seaborn palette.
    Args:
        labels (list): List of labels.
        seaborn_palette (str): Name of the seaborn palette.
    Returns:
        dict: Dictionary color 2 label
    """
    palette = sns.color_palette(seaborn_palette, len(labels)).as_hex()
    return dict(zip(labels, palette))


def explode_fingerpint(session, fingerprint):
    """Explodes the fingerprint into predictions for each labelled statement in the session.
    ########## REPLACED BY 'get_model_prediction' ##########
    Args:
        session (list): The list of statements in the session.
        fingerprint (str): The fingerprint string containing labels and their counts, separated by '--'.
    Returns:
        list: A list of dictionaries representing predictions for each labelled statement. Each dictionary contains:
              - 'label': The label of the statement.
              - 'start': The starting character index of the statement.
              - 'end': The ending character index of the statement.
    """
    # Chek if not nan
    predictions = []
    if fingerprint:
        # Obtain statements
        statements = divide_statements(session=session)
        # Extract items in the shape ["Tactic - statenent_id", "Tactic - statenent_id", ...]
        labels_statement_ids = fingerprint.split(" -- ")
        # Initialize variable to keep track of statements and characters analysed so far
        old_statement_id = 0
        for it_statement, label_statement_id in enumerate(labels_statement_ids):
            label, statement_id = label_statement_id.split(" - ")
            characters_per_statements = 0
            for statement in statements[old_statement_id : int(statement_id) + 1]:
                characters_per_statement = len(statement)
                characters_per_statements += characters_per_statement
            # +1 for each truncated space at the end of each statement (e.g., 'scp -t /tmp/CaW87HUG ;' > missing space at the end)
            characters_per_statements += int(statement_id) + 1 - old_statement_id
            old_statement_id = int(statement_id) + 1
            stopping_character = (
                characters_per_statements - 1
                if it_statement == (len(labels_statement_ids) - 1)
                else characters_per_statements
            )
            predictions.append(
                {
                    "label": label,
                    "start": 0,
                    "end": stopping_character,
                }
            )
    return predictions


def get_model_prediction(model, session):
    """Generates predictions using the specified model on the given session.
    Args:
        model (TokenClassificationPipeline): A pipeline representing the model used for prediction.
        session (object): The input session data for prediction.
    Returns:
        list: A list of dictionaries representing predictions, each containing:
            - 'label' (str): The label or category of the prediction.
            - 'start' (int): The start index of the predicted entity.
            - 'end' (int): The end index of the predicted entity.
            - 'score' (float): The confidence score associated with the prediction.
    """
    predictions = model(session)
    # rename for code compatibility
    renamed_predictions = []
    for prediction in predictions:
        prediction["label"] = prediction.pop("entity_group")
        prediction["end"] = prediction["end"] - prediction["start"]
        prediction["start"] = 0
        prediction["score"] = float(prediction["score"])
        renamed_predictions.append(prediction)
    return renamed_predictions


def extract_fingerprint(output):
    """Extract fingerprint from output.
    This function takes output data containing labeled text items and generates a fingerprint
    representing the number of statements associated with each label.
    Args:
        output (list of dict): A list of dictionaries where each dictionary represents a labeled text item.
            Each dictionary should have keys "label" and "text".
    Returns:
        str: A concatenated string representing the fingerprint, with each item in the format "label - statement_count".
    """
    fingerprint = []
    cumulator_len = 0
    for item in output:
        label = item["label"]
        text = item["text"].strip()
        statements = divide_statements(text)
        fingerprint.append(f"{label} - {len(statements)+cumulator_len}")
        cumulator_len += len(statements)
    return " -- ".join(fingerprint)


def divide_statements(session, add_special_token=False, special_token="[STAT]"):
    """Divide a session into statements.
    This function splits a session into statements using specified separators. Optionally,
    it adds a special token at the beginning of each statement.
    Args:
        session (str): The session to be divided into statements.
        add_special_token (bool): Whether to add a special token to each statement.
        special_token (str, optional): The special token to be added. Defaults to "[STAT]".
    Returns:
        list of str: A list of statements.
    """
    statements = re.split(r"(; |\|\|? |&& )", session + " ")
    # concatenate with separators
    if len(statements) != 1:
        statements = [
            "".join(statements[i : i + 2]).strip()
            for i in range(0, len(statements) - 1, 2)
        ]
    else:  # cases in which there is only 1 statement > must end with " ;"
        statements = [statements[0].strip() + " ;"]
    if add_special_token:
        # Add separator
        statements = [f"{special_token} " + el for el in statements]
    return statements


def extract_file_formatless(file):
    """Function that returns the name of the file without format.
    Args:
        file (str): File with format (e.g., 'sample.json')
    Returns:
        str: filename without format (e.g., 'sample')
    """
    return os.path.splitext(file)[0]


def keep_last_occurrence(fingerprint_str):
    """Keep only the last occurrence of each label in the given fingerprint string.
    Args:
        fingerprint_str (str): The input fingerprint string containing labels and their counts, separated by '--'.
    Returns:
        str: A modified fingerprint string where only the last occurrence of each label is retained.
    """
    segments = fingerprint_str.split(" -- ")
    compact_fingerprint = []
    for segment in segments:
        label, _ = segment.split(" - ")
        last_label = (
            compact_fingerprint[-1].split(" - ")[0]
            if len(compact_fingerprint) != 0
            else None
        )
        if label == last_label:
            # Remove last element
            compact_fingerprint.pop()
            # Append updated
        compact_fingerprint.append(segment)
    return " -- ".join(compact_fingerprint)


def verify_output(fingerprint, session):
    """
    Verifies if all statements in the session have been labelled based on the provided fingerprint.
    Args:
        fingerprint (str): The fingerprint string containing labels and their counts, separated by '--'.
        session (list): The list of statements in the session.
    Returns:
        Flag: If not all statements have been labelled based on the fingerprint.
    """
    statements = divide_statements(session=session)
    last_labelled_statement = int(fingerprint.split(" -- ")[-1].split(" - ")[1])
    return len(statements) == last_labelled_statement
