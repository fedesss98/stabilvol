"""
Creates log to link simulations (MFHTs vs Volatility)
and parameters.
"""
import json
import logging
import datetime
import random
import string

try:
    from stabilvol.utility.definitions import ROOT
except ModuleNotFoundError as e:
    from utility.definitions import ROOT

class Logger:
    def __init__(self):
        # Create random ID
        random_string = str(random.randint(0, 999999)).zfill(6)
        random_letter = random.choice(string.ascii_uppercase)
        self._id = random_string + random_letter

        self.log: dict = {}
        self.inputs: dict = {}
        self.filepath = ROOT / 'data/logs/logs.json'

        logging.info("Logger created")

    def __repr__(self):
        return f"Logger(ID={self.id})"

    def __str__(self):
        return f"{self.id}"

    @property
    def id(self):
        return self._id

    def gather_inputs(self, *classes, **kwargs):
        inputs = {'Date': str(datetime.datetime.now()),
                  **kwargs}
        for c in classes:
            try:
                inputs.update(c.inputs)
            except Exception as e:
                logging.warning("Unrecognized class: ", e)
        self.inputs = inputs
        return self.inputs

    def create_log(self):
        logs_dict = {}
        if self.filepath.exists():
            try:
                # Read existing data and append new
                with open(self.filepath, 'r') as logs_file:
                    logs_dict = json.load(logs_file)
            except json.JSONDecodeError:
                pass
            finally:
                logs_dict.update(self.log)
        else:
            logging.info(f"Logs file not found. Creating a new one in {self.filepath}")
            logs_dict = self.log
        return logs_dict

    def save_log(self, *classes, **kwargs):
        self.gather_inputs(*classes, **kwargs)
        self.log = {self.id: self.inputs}
        # Create new log or update existent log
        log = self.create_log()
        # Overwrite or create log file
        with open(self.filepath, 'w') as logs_file:
            json.dump(log, logs_file, indent=4)
            logging.info(f"Log saved with ID {self.id}")

    def update(self) -> object:
        """ Recreate the random ID """
        random_string = str(random.randint(0, 999999)).zfill(6)
        random_letter = random.choice(string.ascii_uppercase)
        self._id = random_string + random_letter
        return self

    def lookup_logs(self, inputs):
        """
        Check if there is a log with equal content and return the ID of that log
        """
        duplicate_log_id = None
        if self.filepath.exists():
            try:
                # Read existing data if there are
                with open(self.filepath, 'r') as logs_file:
                    logs_dict = json.load(logs_file)
            except json.JSONDecodeError:
                duplicate_log_id = None
            else:
                for log_id in logs_dict.keys():
                    # Check if logs have same parameters
                    if logs_dict[log_id].keys() == inputs.keys():
                        # Check if parameters are equal except for Date
                        if all([logs_dict[log_id][k] == inputs[k] for k in inputs.keys() if k != 'Date']):
                            duplicate_log_id = log_id
        return duplicate_log_id


if __name__ == '__main__':
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter

    meta_data = {
        'Market': 'UN',
    }
    accountant = DataExtractor(start_date='2010-01-01', end_date='2020-01-01')
    analyst = StabilVolter()

    logger = Logger()
    logger.save_log(accountant, analyst, **meta_data)


