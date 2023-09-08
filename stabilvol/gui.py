"""
Graphic User Interface
"""
import threading
from pathlib import Path
from tkinter import filedialog

import seaborn as sns
from pandas import DataFrame

ROOT = Path(__file__).parent

try:
    from stabilvol.utility.definitions import ROOT, MARKETS, MARKETS_STATS
    from stabilvol.utility.styles import StabilvolStyle
    from stabilvol.utility.classes.widgets import *
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter, MeanFirstHittingTimes
    from stabilvol.log.logger import Logger
except ModuleNotFoundError as e:
    from utility.classes.widgets import *
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
        self.progress_window = None
        self.stack.set("Nothing")

        # Objects
        self.datas = []  # Rerturns DataFrame from different markets
        self.logger = Logger()
        self.accountant = DataExtractor(start_date=self.start_date.get(),
                                        end_date=self.end_date.get())
        self.analyst = StabilVolter()
        self.fhts = {}
        self.pdfs = {}
        self.mfhts = {}
        self.log_index = {}
        self.accountant_index = {}

    @property
    def inputs(self):
        return {
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

    @property
    def inputs_iterator(self):
        inputs_to_iterate = [
            self.inputs["markets"],
            self.inputs["start_date"],
            self.inputs["duration"]
        ]
        return itertools.product(*inputs_to_iterate)

    def current_selection(self, total_dict):
        return {key: value for key, value in total_dict.items() if key in self.inputs_iterator}

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
        if len(self.inputs['markets']) <= 0:
            messagebox.showerror(title="No market selected!", message="Select at least one market.")
        else:
            # Initialize DataExtractor
            self.accountant._criterion = self.inputs['criterion']
            self.accountant._value = self.inputs['criterion_value']
            stocks = []
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

    def init_logger(self, **kwargs):
        """
        Initialize the logger and check if inputs have already been used.
        While the analyst is always the same, accountant inputs changes.
        Market name is provided as keyword argument.
        """
        self.logger.update()
        inputs_to_log = self.logger.gather_inputs(self.analyst, self.accountant, **kwargs)
        duplicate_id = self.logger.lookup_logs(inputs_to_log)
        if duplicate_id is not None:
            logging.info(f"Inputs already used in file with ID {duplicate_id}")
            self.logger.id = duplicate_id
        self.results.label_log.configure(text=f"Log ID:\t{self.logger.id}")
        return duplicate_id

    def write_log_indexes(self, market, start, duration):
        """ Create an index to relate a log ID with its distinctive inputs """
        log_index = {
            (market, start, duration): self.logger.id,
        }
        accountant_index = {
            self.logger.id: self.accountant.inputs
        }
        self.log_index.update(log_index)
        self.accountant_index.update(accountant_index)

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
        """ Reformat MFHT as DataFrame and append meta info """
        mfht = mfht.data.to_frame(name="FHT")
        mfht.index.name = "Volatility"
        mfht["Market"] = market
        mfht["Start date"] = start_date
        mfht["Window length"] = int(duration)
        # Save computed mfht in the App
        self.mfhts[(market, start_date, duration)] = mfht

    def start_counting_cycles(self):
        self.progress_window = LoadingWindow(self)
        print("Starting counting cycles")
        threading.Thread(target=self.count_fht).start()

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
            self.init_logger(market=market)
            # Link this ID to its distinctive inputs
            self.write_log_indexes(market, start_date, duration)
            try:
                # Try to load counted FHT from file if this exists
                stabilvol = pd.read_pickle(ROOT / f'data/processed/fht/fht_{self.logger.id}.pickle')
                self.fhts[(market, start_date, duration)] = stabilvol
            except FileNotFoundError:
                # If no file exists, count FHT
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
                self.fhts[(market, start_date, duration)] = stabilvol
            indicators = self.analyst.get_indicators(stabilvol)
            self.print_indicators_table(
                f'{market} {start_date} FHT INDICATORS in {duration} years',
                indicators
            )
            # GET PDF
            pdf: DataFrame = self.analyst.get_pdf(stabilvol)
            self.pdfs[(market, start_date, duration)] = pdf
            # GET MFHT
            # Set maximum volatility to cut the long mfht tail
            max_volatility = indicators['Peak'] + 4*indicators['FWHM']
            mfht = MeanFirstHittingTimes(stabilvol, nbins=inputs['nbins'], max_volatility=max_volatility)
            mfht_indicators = mfht.indicators
            self.print_indicators_table(
                f'{market} {start_date} MFHT INDICATORS in {duration} years',
                mfht_indicators
            )
            self._append_new_mfht(mfht, market, start_date, duration)
        logging.info("Counting terminated.")
        self.progress_window.destroy()
        return None

    def _plot_grid(self, df, plot_type='fht'):
        for d in self.inputs['duration']:
            nrows = len(self.inputs['markets'])
            ncols = len(self.inputs['start_date'])
            fig, axs = plt.subplots(nrows=nrows,
                                    ncols=ncols,
                                    tight_layout=True, figsize=(6 * ncols, 6 * nrows))
            fig.suptitle(f'{d}-years stability analysis', fontsize=20)

            def plot_data(group, duration):
                market = group.name[0]
                start = group.name[1]
                end = (pd.Timestamp('2002') + pd.DateOffset(years=int(duration))).year
                row = self.inputs['markets'].index(market)
                col = self.inputs['start_date'].index(start)
                ax = axs[row] if len(self.inputs['markets']) > 1 else axs
                ax = ax[col] if len(self.inputs['start_date']) > 1 else ax
                self.analyst.plot(plot_type,
                                  data_to_plot=group.reset_index(3), use_ax=ax, title=f'{market} {start}-{end}')
                return None

            data_to_plot = df.loc[(df['Window length'].astype(str) == d)]
            data_to_plot.groupby(['Market', 'Start date'], group_keys=True).apply(plot_data, duration=d)
            plt.show()

    @staticmethod
    def convert_col_name(name):
        if name == 'Start date':
            return 'start_date'
        elif name == 'Window length':
            return 'duration'
        elif name == 'Market':
            return 'markets'
        else:
            return name

    def _plot_stacked(self, df, plot_type='fht'):
        varaibles = ['Market', 'Start date', 'Window length']
        # Take only free variables (not stacked) and their relative input sizes
        variables_len = {v: len(self.inputs[self.convert_col_name(v)]) for v in varaibles if v != self.inputs['stack']}
        # The final plot shoul be "more horizontal" than vertical
        variable_col = max(variables_len, key=variables_len.get)
        variable_row = list(set(varaibles) - {variable_col, self.inputs['stack']})[0]
        plot_params = {
            'x': 'Volatility' if plot_type == 'fht' else 'FHT',
            'y': 'FHT' if plot_type == 'fht' else 'Count',
            'hue': self.inputs['stack'],
            'col': variable_col,
            'row': variable_row
        }
        if plot_type == 'fht':
            g = sns.relplot(data=df, **plot_params)
        elif plot_type == 'pdf':
            # Pass to linearized data in log-log scale
            df = self.analyst.lienarize_powerlaw(df.reset_index(3))
            g = sns.lmplot(data=df, **plot_params)
        # Draws a grid in axes
        if sum(len(self.inputs[self.convert_col_name(v)]) for v in varaibles) == 3:
            g.ax.grid()
        else:
            for ax in g.axes.flat:
                ax.grid()
        plt.show()

    @staticmethod
    def show_selection_error():
        messagebox.showerror(
            "Error", "No data for selected Market, Starting Date or Window Length.")

    @staticmethod
    def take_interval_right_end(df):
        interval = df.index.get_level_values(3)
        if isinstance(interval, pd.IntervalIndex):
            right_end = interval.right
        elif isinstance(interval, pd.CategoricalIndex):
            right_end = interval.categories.right
        else:
            right_end = interval
        df.index = df.index.set_levels(right_end, level=3)
        return df

    def plot_selected_data(self, data, data_type='fht'):
        # Various computed FHTs can be plotted all in separate figures
        # or grouped by market and start date in a grid view
        data_to_plot = None
        try:
            data_to_plot = pd.concat(self.current_selection(data))
        except ValueError:
            self.show_selection_error()
        if self.inputs['stack'] == "Nothing":
            # Group plots by duration and print FHTs across markets on rows and start dates on columns
            self._plot_grid(data_to_plot, plot_type=data_type)
        else:
            self._plot_stacked(data_to_plot, plot_type=data_type)

    def save_analysis(self):
        """ Extract FHT from calculated ones and save them to a pickle file """
        if directory := filedialog.askdirectory(
            title="Please select a directory to save your files"
        ):
            # The user did not cancel the selection
            for market, start_date, duration in self.inputs_iterator:
                id: int = self.log_index[(market, start_date, duration)]
                accountant_inputs = self.accountant_index[id]
                self.logger.id = id
                self.logger.save_log(self.analyst, market=market, **accountant_inputs)
                fht: DataFrame = self.fhts[(market, start_date, duration)]
                fht.to_pickle(Path(directory) / f"fht_{id}.pickle")


def main():
    app = App()
    app.mainloop()
    return None


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('qtagg')
    main()
