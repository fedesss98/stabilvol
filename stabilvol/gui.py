"""
Graphic User Interface
"""
import itertools
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).parent

try:
    from stabilvol.utility.definitions import ROOT, MARKETS, MARKETS_STATS
    from stabilvol.utility.styles import StabilvolStyle
    from stabilvol.utility.classes.widgets import MenuBar, StabilvolFrame, StatusFrame, LeftBar, ButtonFrame, \
        SettingsWindow, ResultsFrame
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter, MeanFirstHittingTimes
    from stabilvol.log.logger import Logger
except ModuleNotFoundError as e:
    from utility.classes.widgets import MenuBar, StabilvolFrame, StatusFrame, LeftBar, ButtonFrame, SettingsWindow, \
        ResultsFrame
    from utility.definitions import ROOT, MARKETS, MARKETS_STATS
    from utility.styles import StabilvolStyle
    from utility.classes.data_extraction import DataExtractor
    from utility.classes.stability_analysis import StabilVolter


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.root = ROOT
        # Icon
        self.iconbitmap(self.root / 'static/icon/stabilvol_128.ico')
        # Style
        self.configure(background='#FFFFFF')
        self.style = StabilvolStyle()
        self.geometry('1290x1000')
        self.resizable(True, True)
        self.minsize(1080, 800)

        self.title("StabilVol")
        self.label = ttk.Label(self, font=self.style.h2,
                               text="Stabilizing Effects of Volatility in Financial Markets",
                               )
        self.label.grid(column=1, row=0, sticky=tk.W, pady=15, padx=5)
        # Configure the grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=4)
        self.rowconfigure(0, weight=2)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=3)
        self.rowconfigure(3, weight=2)
        self.rowconfigure(4, weight=1)

        # Objects
        self.datas = []  # Rerturns DataFrame from different markets
        self.logger = Logger()
        self.accountant = DataExtractor()
        self.analyst = StabilVolter()
        self.fhts = {}
        self.mfhts = {}
        self.log_index = {}

        # VARIABLES
        # Market selection
        self.available_markets = tk.Variable(self, value=list(MARKETS), name='markets')
        self.stocks = tk.Variable(value=[])
        # Data Extraction
        self.start_date = tk.StringVar(self, value='2002', name='start_date')
        self.end_date = tk.StringVar(self, value='2014-01-01', name='end_date')
        self.window_length = tk.StringVar(self, value="6", name='duration')
        self.criterion = tk.StringVar(self, value="StartEnd", name='criterion')
        self.criterion_value = tk.StringVar(self, value="7", name='criterion_value')
        # Stability Analysis
        self.start_level = tk.DoubleVar(self, value=-0.1, name='start_level')
        self.end_level = tk.DoubleVar(self, value=-1.5, name='end_level')
        self.tau_min = tk.IntVar(self, value=2, name='tau_min')
        self.tau_max = tk.IntVar(self, value=10000, name='tau_max')
        # FHT Averaging
        self.nbins = tk.IntVar(self, value=3000, name='nbins')
        # Plotting Options
        self.stack = tk.StringVar(self, value="Nothing", name='stack')
        # Various
        self.save = tk.BooleanVar(self, value=False, name='save')
        self.done_counting = tk.BooleanVar(self, value=False)

        # WIDGETS
        menubar = MenuBar(self)
        self.config(menu=menubar)
        self.marketframe = LeftBar(self)
        self.interface = StabilvolFrame(self)
        self.results = ResultsFrame(self)
        self.buttons = ButtonFrame(self)
        self.statusbar = StatusFrame(self)
        # The setting are closed by default
        self.settings = None

    @property
    def inputs(self):
        inputs_dict = {
            "markets": self.marketframe.marketselection.markets,
            "start_date": self.start_date.get().split(', '),
            "duration": self.window_length.get().split(', '),
            "criterion": self.criterion.get().lower(),
            "criterion_value": self.criterion_value.get(),
            "start_level": self.start_level.get(),
            "end_level": self.end_level.get(),
            "tau_min": self.tau_min.get(),
            "tau_max": self.tau_max.get(),
            "nbins": self.nbins.get(),
            "stack": self.stack.get(),
            "save": self.save.get(),
        }
        return inputs_dict

    @property
    def inputs_iterator(self):
        inputs_to_iterate = [
            self.inputs["markets"],
            self.inputs["start_date"],
            self.inputs["duration"]
        ]
        return itertools.product(*inputs_to_iterate)

    def describe_selected_market(self, event):
        # Show extended name of a market when selected from Listbox
        listbox = event.widget
        listbox_selection = listbox.curselection()
        market = listbox.get(listbox_selection[0])
        market_name = MARKETS_STATS[market][0]
        if len(listbox_selection) > 1:
            market_name += ' et al.'
        self.statusbar.label_market.config(text=f'Selected Market: {market_name}')

    def open_settings(self):
        # Create a new window for the settings
        self.settings = SettingsWindow(self)
        # Start the settings window's event loop
        self.settings.mainloop()

    def select_stocks(self, event=None, show=True):
        # Initialize datas
        self.datas = []
        stocks = []
        if len(self.inputs['markets']) <= 0:
            messagebox.showerror(title="No market selected!", message="Select at least one market.")
        else:
            # Initialize DataExtractor
            self.accountant._criterion = self.inputs['criterion']
            self.accountant._value = self.inputs['criterion_value']
            for market, start_date, duration in self.inputs_iterator:
                print(f'{market} - {start_date} - {duration}')
                # Check if starting date and duration are set up correctly
                # start_date, end_date, duration = self.accountant.check_dates(
                #     start_date, None, duration
                # )
                # Update DataExtractor time window
                self.accountant.start_date = start_date
                self.accountant.duration = duration
                # Extract data
                data = self.accountant.extract_data(self.root / f'data/interim/{market}.pickle')
                stocks.extend(list(data.columns))  # Add market stocks to list
                self.stocks.set(stocks)  # Set stocks TkVriable for the Returnselection Listbox
                self.datas.append(data)
                if show:
                    self.accountant.plot_selection(edit=True).set_title(f"{market} selected stocks")
                    plt.show()
            # Update StatusBar number of stocks
            self.statusbar.label_stocks.config(text=f'Stock included in period: {len(data.columns)} ')
        return None

    def show_returns(self, event):
        stock_selection = int(self.marketframe.returnselection.selected_stock)
        ticker = self.accountant.data.columns[stock_selection]
        self.accountant.plot_returns(ticker=ticker)
        return None

    def init_logger(self, market):
        self.logger.update()
        inputs_to_log = self.logger.gather_inputs(self.accountant, self.analyst, market=market)
        duplicate_id = self.logger.lookup_logs(inputs_to_log)
        if duplicate_id is not None:
            logging.info(f"Inputs already used in file with ID {duplicate_id}")
            self.logger.id = duplicate_id
        self.results.label_log.configure(text=f"Log ID:\t{self.logger.id}")

    def write_log_index(self, market, start, duration):
        """ Create an index to relate a log ID with its distinctive inputs """
        index = {
            (market, start, duration): self.logger.id,
        }
        self.log_index.update(index)

    @staticmethod
    def print_indicators_table(header, indicators):
        logging.info(header)
        keys = list(indicators.keys())
        values = list(indicators.values())
        keys_width = max(len(key) for key in keys)
        values_width = max(len(str(value)) for value in values) + 2
        table_width = keys_width + values_width + 5
        logging.info(f"{'-' * table_width}")
        logging.info(f"|{'Name':^{keys_width}}|{'Value':^{values_width}}|")
        logging.info(f"{'-' * table_width}")
        for key, value in indicators.items():
            logging.info(f"|{key:<{keys_width}}|{str(value):^{values_width}}|")
        logging.info(f"{'-' * table_width}")

    def _append_new_mfht(self, mfht, market, start_date, duration):
        # Take time series values from mfht object and append meta info
        mfht = mfht.data
        mfht["Market"] = market
        mfht["Start date"] = start_date
        mfht["Window length"] = int(duration)
        # Save computed mfht in the App
        self.mfhts[(market, start_date, duration)] = mfht

    def count_fht(self):
        """
        - Retrieve parameters
        - Check if there is already FHT counted with those
        - Count FHT
        :return:
        """
        inputs = self.inputs
        self.analyst.start_level = inputs['start_level']
        self.analyst.end_level = inputs['end_level']
        self.analyst.tau_max = inputs['tau_max']
        for market, start_date, duration in self.inputs_iterator:
            # Set accountant parameters to extract data
            self.accountant.start_date = start_date
            self.accountant.duration = duration
            self.accountant.criterion = (inputs['criterion'], inputs['criterion_value'])
            # Create a new log ID or use that of equal inputs runs
            self.init_logger(market)
            # Link this ID to its distinctive inputs
            self.write_log_index(market, start_date, duration)
            logging.info(f"Run ID: {self.logger.id}")
            # !!! Set sigma_range
            data = self.accountant.extract_data(self.root / f'data/interim/{market}.pickle')
            self.datas.append(data)
            analyst_info = {
                'Market': market,
                'Start date': start_date,
                'Window length': int(duration)
            }
            # GET FHT
            stabilvol = self.analyst.get_stabilvol(data, 'multi', **analyst_info)
            indicators = self.analyst.get_indicators(stabilvol)
            self.print_indicators_table(
                f'{market} {start_date} FHT INDICATORS in {duration} years',
                indicators
            )
            self.fhts[(market, start_date, duration)] = stabilvol
            # Set maximum volatility to cut the long mfht tail
            max_volatility = indicators['Peak'] + 4*indicators['FWHM']
            # GET MFHT
            mfht = MeanFirstHittingTimes(stabilvol, nbins=inputs['nbins'], max_volatility=max_volatility)
            mfht_indicators = mfht.indicators
            self.print_indicators_table(
                f'{market} {start_date} MFHT INDICATORS in {duration} years',
                mfht_indicators
            )
            self._append_new_mfht(mfht, market, start_date, duration)
        logging.info("Counting terminated.")
        return None

    def plot_fht(self):
        # Various computed FHTs can be plotted all in separate figures
        # or grouped by market and start date in a grid view
        fhts = pd.concat(self.fhts)
        if self.inputs['stack'] == "Nothing":
            # Group plots by duration and print FHTs across markets on rows and start dates on columns
            for d in self.inputs['duration']:
                nrows = len(self.inputs['markets'])
                ncols = len(self.inputs['start_date'])
                fig, axs = plt.subplots(nrows=nrows,
                                        ncols=ncols,
                                        tight_layout=True, figsize=(6*ncols, 6*nrows))
                fig.suptitle(f'{d}-years stability analysis', fontsize=20)

                def plot_data(group, duration):
                    market = group.name[0]
                    start = group.name[1]
                    end = (pd.Timestamp('2002')+pd.DateOffset(years=int(duration))).year
                    row = self.inputs['markets'].index(market)
                    col = self.inputs['start_date'].index(start)
                    ax = axs[row] if len(self.inputs['markets']) > 1 else axs
                    ax = ax[col] if len(self.inputs['start_date']) > 1 else ax
                    self.analyst.plot_fht(group, use_ax=ax,
                                          title=f'{market} {start}-{end}')
                    return None

                data_to_plot = fhts.loc[fhts['Window length'] == int(d)]
                data_to_plot.groupby(['Market', 'Start date'], group_keys=True).apply(plot_data, duration=d)
                plt.show()
        else:
            free_variables = [c for c in ['Market', 'Start date', 'Window length'] if c != self.inputs['stack']]
            g = sns.relplot(data=fhts,
                            x='Volatility',
                            y='FHT',
                            hue=self.inputs['stack'],
                            row=free_variables[0],
                            col=free_variables[1],)
            plt.show()

    def plot_mfht(self):
        # Various computed FHTs can be plotted all in separate figures
        # or grouped by market and start date in a grid view
        fhts = pd.concat(self.fhts)
        if self.inputs['stack']:
            for d in self.inputs['duration']:
                fig, axs = plt.subplots(nrows=len(self.inputs['markets']),
                                        ncols=len(self.inputs['start_date']),
                                        tight_layout=True)
                fig.suptitle(f'{d}-years stability analysis')

                def plot_data(group):
                    market = group.name[0]
                    start = group.name[1]
                    row = self.inputs['markets'].index(market)
                    col = self.inputs['start_date'].index(start)
                    sns.scatterplot(group,
                                    x='Volatility',
                                    y='FHT',
                                    ax=axs[row][col])

                data_to_plot = fhts.loc[fhts['Window length'] == d]
                data_to_plot.groupby(['Market', 'Start date']).apply(plot_data)

    def save_analysis(self):
        directory = filedialog.askdirectory(title="Please select a directory to save your files")
        if directory:
            # The user did not cancel the selection
            for market, start_date, duration in self.inputs_iterator:
                id: int = self.log_index[(market, start_date, duration)]
                self.logger.save_log(self.analyst, market=market, start_date=start_date, duration=duration)
                fht: pd.DataFrame = self.fhts[(market, start_date, duration)]
                mfht: pd.DataFrame = self.mfhts[(market, start_date, duration)]
                fht.to_pickle(Path(directory) / f"fht_{id}.pickle")
                mfht.to_pickle(Path(directory) / f"mfht_{id}.pickle")

def main():
    app = App()
    app.mainloop()
    return None


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('qtagg')
    main()
