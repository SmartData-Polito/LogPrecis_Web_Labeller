"""Web Labeller app for LogPrecis's labels. 
The user gets as input a SSH attack and has to classify it's intents.
"""

import json
import os
from sys import argv
from flask import Flask, redirect, render_template, request, url_for
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from sub.utils import (
    get_model_prediction,
    extract_fingerprint,
    read_json_file,
    load_data,
    save_data,
    save_json,
    get_labels_2_colors,
    extract_file_formatless,
    keep_last_occurrence,
    verify_output,
)


class WebApp:
    """Class to label an unlabelled dataset using a Flask interface.
    The script will:
    - Load an unlabelled dataset from an 'input_path'
    - Load LogPrecis to get the real-time predictions for each session
    - Present the labeller a web-interface to correct the predictions and produce a human-curated label
    - Save the results into an 'output_path'

    The output path will act as a key for continuing the labelling if interrupted earlier.
        This means: if we were labelling some data and stopped, we stored the last labelled index and continue from there.
    N.b: for the indexes, ORDER MATTERS (index is obtained via .iloc) > do not shuffle the same input dataset, otherwise you lose the state!
    """

    def __init__(
        self,
        input_path,
        input_file,
        output_path,
        output_file,
        path_main_html,
        labels,
        seaborn_palette="hls",
        model_name="SmartDataPolito/logprecis",
        device="cpu",
    ):
        """Initialize the LabellingSystem instance.
        Args:
            input_path (str): The path to the input data.
            input_file (str): The filename of the input data.
            output_path (str): The path to save the output data.
            output_file (str): The filename to save the output data.
            path_main_html (str): The path to the main HTML file for web interface.
            labels (list): A list of labels for classification.
            seaborn_palette (str, optional): The seaborn color palette name for label colors.
                Defaults to "hls".
            model_name (str, optional): The name of the model for prediction. Update if you want to use an updated version.
                Defaults to "SmartDataPolito/logprecis".
            device (int/str, optional): The device number for model inference (e.g., 0 for "cuda:0") or "cpu".
                Defaults to "cpu".
        """
        # 1. Read the data
        self.unlabelled_df = self.read_data(path=input_path, file=input_file)
        # 2. Create Flask App (i.e., making the interface start).
        self.app = self.create_app()
        # 3. Name of output columns is columns of input dataframe + the labeller's fingerprint
        # 3.a Make sure, in case the dataset is weakly supervised, to not consider the fingerprint twice
        df_columns = [
            el for el in list(self.unlabelled_df.columns) if el != "fingerprint"
        ]
        # 3.b. append the fingerprint to the columns we will store (fingerprint given/validated by this script).
        self.column_names = df_columns + ["fingerprint"]
        # 4. Read current output and check if we restart a stopped labelling process
        #   N.B. new labelled elements (restarting old stopped process) will be appended here!
        self.old_output_data, self.info_previous_runs = self.read_info_previous_runs(
            path=output_path, output_filename=output_file
        )
        # 5. extract current index we want to start labelling from
        self.current_index = self.info_previous_runs["last_labelled_index"] + 1
        # 6. Get pages we want to deploy on the web interface
        self.path_main_html = path_main_html
        # 7. Register the routes (start web server)
        self.register_routes()
        # 8. Dummy initialization for operational variables
        self.current_output = []
        self.status_current_labelling = {
            "skipped_indexes": [],
            "last_labelled_index": self.current_index,
        }
        # 9. Save output path and filename for later
        self.output_path, self.output_file = output_path, output_file
        # 10. Prepare mapping from labels 2 colors
        self.labels2colors = get_labels_2_colors(
            labels=labels, seaborn_palette=seaborn_palette
        )
        # 11. Load model to make predictions on the unlabelled corpus
        #   N.b. Notice: the model's predictions will serve as 'soft labels' for the labeller
        self.logprecis = self.load_prediction_pipeline(
            model_name=model_name, device=device
        )

    def read_data(self, path, file):
        """Function that reads and process an input DataFrame from a configuration file.
        Args:
            path (str): input path
            file (str): name of input file
        Returns:
            DataFrame: dataframe containing the data to label.
        """
        # 0. Check that data exists
        assert os.path.isfile(os.path.join(path, file)), "Error: input file must exist"
        # 1. Load the data
        df = load_data(path=path, input_file=file)
        # 2. Preprocess
        #   2a. Drop NaNs
        df = df.dropna(axis=0)
        #   2b. Remove the "fingerprint" column if exists
        if "fingerprint" in df.columns:
            df.drop(["fingerprint"], axis=1, inplace=True)
        return df

    def create_app(self):
        """
        In the context of a Flask application, __name__ is typically
        used to help Flask determine the root path of the application
        and locate the resources (templates, static files, etc.) relative to that path.
        """
        app = Flask(__name__)
        app.config["SERVER_SHUTDOWN"] = True
        return app

    def read_info_previous_runs(self, path, output_filename):
        """Function that checks if there was a previous run with the same output_filename.
            If it is the case, we want to keep labelling from the previous stopping point and update the old labels.
        Args:
            path (str): Path to results
            output_filename (str): Name of the output file.
                If there are indexes to skip, by construction a
                file f"skipped_indexes_{output_filename}" exists
        Returns:
            (DataFrame, dict): Dictionary with the info about the stopped run (e.g., 'skipped_indexes' and 'last_labelled_index')
                                and previously labelled data.
        """
        # 1. Load previous labelled sessions
        previous_output_df = self.read_previous_output(path=path, file=output_filename)
        assert (
            list(previous_output_df.columns) == self.column_names
        ), "Error: the new input dataframe must have the same columns as the ones from previous runs"
        # 2. Load parameters about old runs
        retrieved_parameters = self.retrieve_old_parameters(
            path=path, file=output_filename
        )
        return previous_output_df, retrieved_parameters

    def retrieve_old_parameters(self, path, file):
        """This function returns the parameters about previous runs or an empty dictionary
        Args:
            path (str): input path
            file (str): name of input file
        Returns:
            dict: Dictionary with the info about the stopped run (e.g., 'skipped_indexes' and 'last_labelled_index').
        """
        # N.b. file act as a key to retrieve old settings
        formatless_file = extract_file_formatless(file)
        file_path = os.path.join(path, f"params_old_runs_{formatless_file}.json")
        if os.path.isfile(file_path):
            retrieved_parameters = read_json_file(filepath=file_path)
        else:
            retrieved_parameters = {"skipped_indexes": [], "last_labelled_index": 0}
        return retrieved_parameters

    def read_previous_output(self, path, file):
        """This function returns the previous labelled data (if any) or an empty dataframe.
        Args:
            path (str): input path
            file (str): name of input file
        Returns:
            DataFrame: Previous labelled data.
        """
        if os.path.isfile(os.path.join(path, file)):
            output_df = self.read_data(path, file)
        else:
            output_df = pd.DataFrame([], columns=self.column_names)
        return output_df

    def register_routes(self):
        """Function that rules the behaviour of the web interface + how to update the current output when the labeller makes choice."""

        def index():
            # 1. Check if we still have data to
            if self.current_index >= len(self.unlabelled_df):
                # If not, close the project
                return terminate_interface()
            # 2. If we have, keep labelling from current position
            unlabelled_session = self.unlabelled_df.iloc[self.current_index]["session"]
            # 3. Extract oracle's predictions, if any
            predictions = get_model_prediction(
                model=self.logprecis,
                session=self.unlabelled_df.iloc[self.current_index]["session"],
            )
            # 4. Render the template
            return render_template(
                template_name_or_list=self.path_main_html,
                unlabelled_session=unlabelled_session,
                remaining=len(self.unlabelled_df) - self.current_index,
                predictions=json.dumps(predictions),
                id=self.current_index,
                class_colors=self.labels2colors,
            )

        self.app.add_url_rule("/", "index", index)

        def skip():
            # If the session is skipped, save the skipped index
            self.status_current_labelling["skipped_indexes"].append(self.current_index)
            self.current_index += 1
            return redirect(url_for("index", index=self.current_index))

        self.app.add_url_rule("/skip", "skip", skip, methods=["POST"])

        def continue_labelling():
            # Receive labeled data from the frontend
            labeled_data = request.json
            self.process_output(output=labeled_data)
            self.current_index += 1
            self.status_current_labelling["last_labelled_index"] = self.current_index
            return redirect(url_for("index"))

        self.app.add_url_rule(
            "/continue", "continue", continue_labelling, methods=["POST"]
        )

        def terminate_interface():
            self.save_current_results()
            # Implement saving stored values to a new dataframe and conclude the script
            return render_template(
                "conclusion.html",
                remaining=len(self.unlabelled_df) - self.current_index,
            )

        self.app.add_url_rule("/exit", "exit", terminate_interface, methods=["POST"])

    def load_prediction_pipeline(self, model_name, device):
        """Function that loads a token classification pipeline.
        Script will call pipeline(session) to get the model's predictions.
        Args:
            model_name (str): Name of the model
            device (int/str): Cuda device number / cpu.
        Returns:
            TokenClassificationPipeline: a token classification pipeline
        """
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = f"cuda:{device}" if isinstance(device, int) else "cpu"
        # For 'aggregation strategy':
        # See https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/pipelines#transformers.TokenClassificationPipeline
        return pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy="simple",
        )

    def process_output(self, output):
        """Process the output data and append results to current output.
        This method processes the output data by extracting a fingerprint, printing it,
        and appending the results to the current output list.
        Args:
            output (list of dict): A list of dictionaries representing labeled text items.
        """
        fingerprint = keep_last_occurrence(extract_fingerprint(output=output))
        if not verify_output(
            fingerprint=fingerprint,
            session=self.unlabelled_df.iloc[self.current_index]["session"],
        ):
            self.save_current_results()
            raise Exception("Error: not all statements have been labelled!")
        results = []
        column_names = [
            column for column in self.column_names if column != "fingerprint"
        ]
        for column in column_names:
            results.append(self.unlabelled_df.iloc[self.current_index][column])
        results.append(fingerprint)
        self.current_output.append(results)

    def save_current_results(self):
        """Save the current results to output files and update configuration for the next run."""
        # 1. Concatenate current outputs into a dictionary
        current_output = pd.DataFrame(self.current_output, columns=self.column_names)
        # 2. Concatenate with former results, if any
        output_df = (
            pd.concat([self.old_output_data, current_output])
            if len(self.old_output_data) > 0
            else current_output.copy()
        )
        # 3. Save new output file
        save_data(df=output_df, output_path=self.output_path, file=self.output_file)
        # 4. Also saving the current labelling status for the next experiments
        #   4.1 Update old status with new
        updated_status = self.update_labelling_status(
            status_current_labelling=self.status_current_labelling,
            info_previous_runs=self.info_previous_runs,
        )
        #   4.2 Actual saving
        save_json(
            dictionary=updated_status,
            output_path=self.output_path,
            file=f"params_old_runs_{self.output_file}",
        )

    def update_labelling_status(self, status_current_labelling, info_previous_runs):
        """Function that joins the previous and current labelling status.
        Args:
            status_current_labelling (dict): Metadata about current labelling process.
            info_previous_runs (dict): Metadata with aggregated results from previous runs
        Returns:
            dict: Updated status (updating the list of skipped indexes)
        """
        # Last index not labelled yet
        updated_status = {
            "skipped_indexes": info_previous_runs["skipped_indexes"]
            + status_current_labelling["skipped_indexes"],
            "last_labelled_index": status_current_labelling["last_labelled_index"] - 1,
        }
        return updated_status

    def run_service(self, debug, port_number):
        """Run the service.
        Args:
            debug (bool): Flag indicating whether to run the service in debug mode.
            port_number (int): port in which we expose the service.
        """
        self.app.run(debug=debug, port=port_number)


if __name__ == "__main__":
    assert len(argv) == 2, "Error: script requires a config file with parameters!"
    CONFIG_PATH = argv[1]
    CONFIG = read_json_file(filepath=CONFIG_PATH)
    web_app = WebApp(
        input_path=CONFIG["input_path"],
        input_file=CONFIG["input_file"],
        output_path=CONFIG["output_path"],
        output_file=CONFIG["output_file"],
        path_main_html=CONFIG["main_html"],
        labels=CONFIG["labels"],
    )
    web_app.run_service(debug=True, port_number=CONFIG["port_number"])
