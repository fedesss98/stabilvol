"""
Creates log to link simulations (MFHTs vs Volatility)
and parameters.
"""
import json
import logging
import datetime
import random
import string


class Logger(dict):
    def __init__(self):
        self.log: dict = {}
        self.inputs: dict = {}
        self.id: str = '000000A'
        logging.info("Logger created")

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

    def save_log(self, *classes, **kwargs):
        random_string = str(random.randint(0, 999999)).zfill(6)
        random_letter = random.choice(string.ascii_uppercase)
        self.id = random_string + random_letter
        self.gather_inputs(*classes, **kwargs)
        self.log = {self.id: self.inputs}
        with open("logs.json", "a+") as logs_file:
            json.dump(self.log, logs_file)


if __name__ == '__main__':
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter

    meta_data = {
        'Market': 'UN',
    }
    accountant = DataExtractor()
    analyst = StabilVolter()

    logger = Logger()
    logger.save_log(accountant, analyst, **meta_data)


