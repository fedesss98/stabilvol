"""
Different GUI for different applications.

GUIs are Tkinter objects.
"""
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import logging
import pandas as pd
import itertools

try:
    from stabilvol.utility.definitions import ROOT
    from stabilvol.utility.styles import FONTS
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter
    from stabilvol.analyze.count_fht import main as count_fht
except ModuleNotFoundError as e:
    logging.warning(f"Error in count_fht imports: {e}")
    from utility.definitions import ROOT
    from utility.styles import FONTS
    from utility.classes.data_extraction import DataExtractor
    from utility.classes.stability_analysis import StabilVolter
    from analyze.count_fht import main as count_fht

from pathlib import Path

DATABASE = Path(__file__).parent.parent.parent.parent / 'data'
MARKETS = [file.stem for file in (DATABASE / 'interim').glob('*.pickle')]


class MarketSelectionFrame(ttk.Frame):
    """
    Market selection Frame.
    You can choose one or more market from MARKETS to analyze.
    """

    def __init__(self, container, col=1, row=1, rowspan=1):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Markets", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self,
            text="Select one or more Markets from the list",
            font=FONTS['p-help'],
        )

        self._markets = tk.Variable(value=list(MARKETS))
        self.markets_list = tk.Listbox(
            self,
            listvariable=container.available_markets,
            height=len(MARKETS),
            selectmode=tk.EXTENDED,
            exportselection=0,
            font=FONTS['p'],
        )
        # Colorize alternating lines of the listbox
        for i in range(0, len(MARKETS), 2):
            self.markets_list.itemconfigure(i, background='#f0f0ff')

        # Binding market data extraction to left mouse double click
        self.markets_list.bind('<Double-Button-1>', container.select_stocks)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.markets_list.grid(column=0, row=2, sticky=tk.EW)
        # Place Frame.
        self.grid(column=col, row=row, rowspan=rowspan, padx=5, pady=5, sticky=tk.W)

    @property
    def markets(self):
        return [self.markets_list.get(i) for i in self.markets_list.curselection()]


class WindowsTextFrame(ttk.Frame):
    """
    Windows selection Frame.
    You can enter the lengths of the windows to use.
    """

    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Windows", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Select the lengths of the Windows, separated by a comma.",
        )
        self.window_lengths = tk.Entry(self, textvariable=container.window_length)
        # self.window_lengths.insert('1.0', "10")
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.window_lengths.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class StartTextFrame(ttk.Frame):
    """
    Frame to select the starting date for the Window.
    """

    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Start Date", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Insert the starting date for the windows (YYYY-MM-DD)",
        )
        self.start_string = tk.Entry(self, textvariable=container.start_date, width=15)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.start_string.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)

    @property
    def start_date(self):
        return self.start_string.get()


class EndTextFrame(ttk.Frame):
    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="End Date", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self,
            text="Insert the ending date for the windows (YYYY-MM-DD or YYYY)",
            font=FONTS['p-help'],
        )
        self.start_string = tk.Entry(self, textvariable=container.end_date)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.start_string.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class CriterionDropdownFrame(ttk.Frame):
    """
    Criterion selection Frame.
    You can choose one criterion between 'percentage' and 'threshold' and add a value.
    """

    def __init__(self, container, col, row):
        super().__init__(container)

        self.container = container

        self.frame_label = ttk.Label(self, text="Criterion", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=280,
            text="Choose one selection criterion for returns "
                 "(Float for Percentage and int number of days for StartEnd)",
        )
        # Help with the criterion value
        self._value_label_help = tk.StringVar(self, value='Days')
        self.value_label_help = ttk.Label(
            self, font=FONTS['p-help'], textvariable=self._value_label_help
        )
        self.criterion_selection = ttk.Combobox(self,
                                                state='readonly',
                                                values=["Percentage", "StartEnd"],
                                                textvariable=container.criterion)
        self.value_input = tk.Entry(self, textvariable=container.criterion_value, width=10)
        # Bind change in combox with change in help for criterion value
        self.criterion_selection.bind('<<ComboboxSelected>>', self.update_help)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, columnspan=2, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, columnspan=2, sticky=tk.W)
        self.criterion_selection.grid(column=0, row=2, columnspan=2, sticky=tk.NW)
        self.value_label_help.grid(column=0, row=3, sticky=tk.NW, pady=4)
        self.value_input.grid(column=1, row=3, sticky=tk.NW, pady=4)
        # Place Frame
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)

    def update_help(self, event):
        if self.container.criterion.get() == "StartEnd":
            self._value_label_help.set("Days")
        elif self.container.criterion.get() == "Percentage":
            self._value_label_help.set("Fraction")


class SlideChoiceFrame(ttk.Frame):
    """
    Windows selection Frame.
    You can enter the lengths of the windows to use.
    """

    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Slide", font=FONTS['h4'])
        self._slide_choice = tk.StringVar()
        self.yes_choice = ttk.Radiobutton(self, text="Yes", value="True", variable=self._slide_choice,
                                          command=self.manage_end_input)
        self.no_choice = ttk.Radiobutton(self, text="No", value="False", variable=self._slide_choice,
                                         command=self.manage_end_input)
        self._slide_choice.set("False")

        self.end_string_label = ttk.Label(self, text="Until:", font=FONTS['p-help'])
        self.end_string = tk.Text(self, height=1, width=20)
        self.end_string.insert('1.0', "---")
        self.end_string.config(state=tk.DISABLED, background='gray95')
        # Place Widgets

        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.yes_choice.grid(column=0, row=1, sticky=tk.EW)
        self.no_choice.grid(column=0, row=2, sticky=tk.EW)
        self.end_string_label.grid(column=0, row=3, sticky=tk.W)
        self.end_string.grid(column=0, row=4, sticky=tk.W)
        # Place Frame
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.EW)

    @property
    def choice(self):
        return self._slide_choice.get()

    @property
    def end_date(self):
        if self.choice == "True":
            return self.end_string.get(1.0, 'end-1c')
        else:
            return None

    def manage_end_input(self):
        if self._slide_choice.get() == "True":
            self.end_string.config(state=tk.NORMAL, background='white')
            self.end_string.delete('1.0', tk.END)
            self.end_string.insert('1.0', "2022-07-01")
        else:
            self.end_string.delete('1.0', tk.END)
            self.end_string.insert('1.0', "---")
            self.end_string.config(state=tk.DISABLED, background='gray95')


class SaveChoiceFrame(ttk.Frame):
    """
    Windows selection Frame.
    You can enter the lengths of the windows to use.
    """

    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Save", font=FONTS['h4'])
        self._save_choice = tk.BooleanVar(self, value=False)
        self.yes_choice = ttk.Radiobutton(self, text="Yes", value=True, variable=container.save)
        self.no_choice = ttk.Radiobutton(self, text="No", value=False, variable=container.save)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.yes_choice.grid(column=0, row=1, sticky=tk.EW)
        self.no_choice.grid(column=0, row=2, sticky=tk.EW)
        # Place Frame
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.EW)

    @property
    def choice(self):
        return self._save_choice.get()


class StartLevelTextFrame(ttk.Frame):
    """
    Frame to select the starting threshold for FHT counting.
    """

    def __init__(self, container, col, row, variable):
        super().__init__(container)
        self.variable = variable if variable else tk.StringVar()
        self.frame_label = ttk.Label(self, text="Start Level: ", font=FONTS['p-help'])
        self.start_string = tk.Entry(self, textvariable=variable, width=8)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.start_string.grid(column=1, row=0, sticky=tk.W, padx=2)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class EndLevelTextFrame(ttk.Frame):
    """
    Frame to select the ending threshold for FHT counting.
    """

    def __init__(self, container, col, row, variable=None):
        super().__init__(container)
        self.variable = variable if variable else tk.StringVar()
        self.frame_label = ttk.Label(self, text="End Level:  ", font=FONTS['p-help'])
        self.end_string = tk.Entry(self, textvariable=self.variable, width=8)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.end_string.grid(column=1, row=0, sticky=tk.W, padx=2)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class ThresholdFrame(ttk.Frame):
    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Thresholds", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Insert threshold levels for FHT counting",
        )
        # Place widgets
        self.frame_label.grid(column=0, row=0, columnspan=2, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, columnspan=2, sticky=tk.W)
        self.start = StartLevelTextFrame(self, col=0, row=2, variable=container.start_level)
        self.end = EndLevelTextFrame(self, col=0, row=3, variable=container.end_level)
        # Place Frame
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class NbinsFrame(ttk.Frame):
    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Number of bins", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Insert the number of bins in which to average the FHT",
        )
        self.start_string = tk.Entry(self, textvariable=container.nbins)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.start_string.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class StackChoiceFrame(ttk.Frame):
    """
    Windows selection Frame.
    You can enter the lengths of the windows to use.
    """

    def __init__(self, container, col, row):
        super().__init__(container)
        # Insert separator before widget
        ttk.Separator(self, orient=tk.HORIZONTAL).grid(column=0, row=0, ipadx=150, pady=10)

        self.frame_label = ttk.Label(self, text="Stack plots", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Stack multiple plots on the same chart or view them separately",
        )
        self.yes_choice = ttk.Radiobutton(self, text="Yes", value=True, variable=container.stack)
        self.no_choice = ttk.Radiobutton(self, text="No", value=False, variable=container.stack)
        # Place Widgets
        self.frame_label.grid(column=0, row=1, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=2, sticky=tk.W)
        self.yes_choice.grid(column=0, row=3, sticky=tk.EW)
        self.no_choice.grid(column=0, row=4, sticky=tk.EW)
        # Place Frame
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.EW)


class ReturnsSelection(ttk.Frame):
    """
    Returns selection Frame.
    Display stocks in selected market.
    """

    def __init__(self, container, col=1, row=1, rowspan=1):
        super().__init__(container)

        self.container = container

        self.frame_label = ttk.Label(self, text="Stocks", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self,
            text="Stocks in selected market.\nDouble click to show returns",
            font=FONTS['p-help'],
        )
        self.stocks_list = tk.Listbox(
            self,
            listvariable=container.stocks,
            height=15,
            selectmode=tk.BROWSE,
            exportselection=0,
            font=FONTS['p'],
        )
        # Binding double click left mouse
        self.stocks_list.bind('<Double-Button-1>', container.show_returns)

        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.stocks_list.grid(column=0, row=2, sticky=tk.EW)
        # Place Frame.
        self.grid(column=col, row=row, rowspan=rowspan, padx=5, pady=5, sticky=tk.W)

    @property
    def selected_market(self):
        if len(self.container.marketselection.markets) > 0:
            return self.container.marketselection.markets[0]
        else:
            return None


class StatusFrame(ttk.Frame):
    """
    Show status of the app:
    - number of stocks selected
    - std of the market
    - counted/not counted fht
    """

    def __init__(self, container, col, row, columnspan=1):
        super().__init__(container)

        self.container = container

        # Labels
        self.l_number = ttk.Label(textvariable=self.container.done_counting)

        # Place labels
        self.l_number.grid(column=0, row=0, pady=5, padx=2)

        # Place frame
        self.grid(column=col, row=row, columnspan=columnspan, padx=20, pady=10, sticky=tk.NSEW)


class StabilvolFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        self.stabilvols_binned = []
        self.container = container
        self.done_counting = tk.BooleanVar(self, value=False)
        # Variables
        self.available_markets = tk.Variable(self, value=MARKETS, name='markets')
        self.start_date = tk.StringVar(self, value='2002', name='start_date')
        self.end_date = tk.StringVar(self, value='2014-01-01', name='end_date')
        self.window_length = tk.StringVar(self, value="6", name='duration')
        self.criterion = tk.StringVar(self, value="StartEnd", name='criterion')
        self.criterion_value = tk.StringVar(self, value="7", name='criterion_value')
        self.start_level = tk.DoubleVar(self, value=-0.1, name='start_level')
        self.end_level = tk.DoubleVar(self, value=-1.5, name='end_level')
        self.tau_min = tk.IntVar(self, value=2, name='tau_min')
        self.tau_max = tk.IntVar(self, value=10000, name='tau_max')
        self.nbins = tk.IntVar(self, value=3000, name='nbins')
        self.stack = tk.BooleanVar(self, value=False, name='stack')
        self.save = tk.BooleanVar(self, value=False, name='save')

        # Objects
        self.accountant = DataExtractor()
        self.stocks = tk.Variable(value=[])
        self.datas = []
        self.analyst = StabilVolter()
        self.stabilvols = []
        self.mfhts = []

        # Fields options
        options = {'padx': 5, 'pady': 5}

        # Label
        self.label = ttk.Label(self, text='First Hitting Times Counting', font=FONTS['h3'])
        self.label.grid(column=2, row=0, columnspan=2, sticky=tk.NW, **options)

        # Widgets
        # Col 1
        self.marketselection = MarketSelectionFrame(self, col=0, row=0, rowspan=2)
        self.returnselection = ReturnsSelection(self, col=0, row=2, rowspan=7)
        ttk.Separator(self, orient=tk.VERTICAL).grid(column=1, row=0, rowspan=8, sticky=tk.NS, padx=40)
        # Col 2
        self.startinput = StartTextFrame(self, col=2, row=1)
        self.windowsinputs = WindowsTextFrame(self, col=2, row=2)
        self.criterioniputs = CriterionDropdownFrame(self, col=2, row=3)
        # Col 3
        self.thresholdsinputs = ThresholdFrame(self, col=3, row=1)
        self.nbinsinput = NbinsFrame(self, col=3, row=2)
        self.stackinput = StackChoiceFrame(self, col=3, row=3)

        # Buttons
        # Show selected stocks
        self.btn_select = ttk.Button(self, text='Show selected stocks', width=10, command=self.select_stocks)
        self.btn_select.grid(column=2, row=4, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        # Count FHT
        self.btn_count = ttk.Button(self, text='Count!', width=10, command=self.start_counting)
        self.btn_count.grid(column=2, row=5, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        # Show FHT
        self.btn_fht = ttk.Button(self, text='Show FHT', width=10, command=self.show_fht, state=tk.DISABLED)
        self.btn_fht.grid(column=2, row=6, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        # Show MFHT
        self.btn_mfht = ttk.Button(self, text='Show MFHT', width=10, command=self.show_mfht, state=tk.DISABLED)
        self.btn_mfht.grid(column=2, row=7, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        # Save series
        self.btn_save = ttk.Button(self, text='Save series to csv', width=10, command=self.save_series,
                                   state=tk.DISABLED)
        self.btn_save.grid(column=2, row=8, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        # Inputs frame
        self.inputs_frame = ttk.LabelFrame(self, text="Inputs", width=600)
        self.inputs_label = ttk.Label(self.inputs_frame, text="Inputs will be displayed here")
        self.inputs_frame.grid(column=0, row=9, columnspan=4, sticky=tk.NSEW, **options)
        self.inputs_label.grid(row=0, column=0, columnspan=2)

        # Infobar frame
        self.status = StatusFrame(self, col=0, row=10, columnspan=4)

        # add padding to the frame and show it
        self.grid(column=0, row=1, padx=20, pady=10, sticky=tk.NSEW)

    @property
    def inputs(self):
        inputs_dict = {
            "markets": self.marketselection.markets,
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
    def input_iterator(self):
        inputs_to_iterate = [
            self.inputs["markets"],
            self.inputs["start_date"],
            self.inputs["duration"]
        ]
        return itertools.product(*inputs_to_iterate)

    @property
    def multiple_markets(self):
        return len(self.inputs['markets']) > 1

    @property
    def multiple_starts(self):
        return len(self.inputs['start_date']) > 1

    @property
    def multiple_durations(self):
        return len(self.inputs['duration']) > 1

    def select_stocks(self, event=None, show=True):
        # Initialize datas
        self.datas = []
        stocks = []
        # With new selection there is no FHT or MFHT to show,
        # so disable showing buttons
        self.btn_fht['state'] = tk.DISABLED
        self.btn_mfht['state'] = tk.DISABLED
        self.btn_save['state'] = tk.DISABLED
        if len(self.inputs['markets']) <= 0:
            messagebox.showerror(title="No market selected!", message="Select at least one market.")
        else:
            # Initialize DataExtractor
            self.accountant._criterion = self.inputs['criterion']
            self.accountant._value = self.inputs['criterion_value']
            for market, start_date, duration in self.input_iterator:
                print(f'{market} - {start_date} - {duration}')
                # Check if starting date and duration are set up correctly
                start_date, end_date, duration = self.accountant.check_dates(
                    start_date, None, duration
                )
                # Update DataExtractor time window
                self.accountant.start_date = start_date
                self.accountant.end_date = end_date
                self.accountant.duration = duration
                # Extract data
                data = self.accountant.extract_data(self.container.root / f'data/interim/{market}.pickle')
                stocks.extend(list(data.columns))  # Add market stocks to list
                self.stocks.set(stocks)  # Set stocks variable for the returnselection Listbox
                self.datas.append(data)
                if show:
                    self.accountant.plot_selection(edit=True).set_title(f"{market} selected stocks")
                    plt.show()
        return None

    def show_returns(self, event):
        stock_selection = int(self.returnselection.stocks_list.curselection()[0])
        ticker = self.accountant.data.columns[stock_selection]
        self.accountant.plot_returns(ticker=ticker)
        return None

    def updates_input_label(self):
        for row, input in enumerate(self.inputs.keys()):
            value = self.inputs.get(input)
            input_label = ttk.Label(self.inputs_frame, text=input, font='Arial 10 bold', width=10, anchor='w')
            input_label.grid(row=row, column=0)
            value_label = ttk.Label(self.inputs_frame, text=value, font=('Lucida Console', 10), width=10, anchor='e')
            value_label.grid(row=row, column=1)
        self.done_counting.set(True)
        return None

    def start_counting(self):
        """
        Count FHT and make data available in list self.stabilvols.
        """
        self.stabilvols = []
        self.stabilvols_binned = []
        self.updates_input_label()
        self.analyst.start_level = self.inputs['start_level']
        self.analyst.end_level = self.inputs['end_level']
        self.analyst.tau_min = self.inputs['tau_min']
        self.analyst.tau_max = self.inputs['tau_max']
        if len(self.inputs['markets']) <= 0:
            messagebox.showerror(title="No market selected!", message="Select at least one market.")
        else:
            if len(self.datas) == 0:
                # Datas are not extracted yet,
                # extract but don't show result
                self.select_stocks(show=False)
            for i, inputs in enumerate(self.input_iterator):
                market = inputs[0]
                start = inputs[1]
                duration = inputs[2]
                stabilvol = self.analyst.get_stabilvol(self.datas[i])
                stabilvol['Market'] = market
                stabilvol['Start date'] = start
                stabilvol['Window length'] = duration
                self.stabilvols.append(stabilvol)
            # Enable buttons to show/save results
            self.btn_fht['state'] = tk.NORMAL
            self.btn_mfht['state'] = tk.NORMAL
            # Trah extracted data
            self.datas = []
            print("Countitng finished. You can visualize results now.")
        return None

    def show_fht(self, stack=False):
        for i, inputs in enumerate(self.input_iterator):
            market = inputs[0]
            start_date = inputs[1]
            duration = inputs[2]
            fht = self.stabilvols[i]
            self.analyst.plot_fht(fht, title=f"{market} FHT from {start_date} for {duration} years")
        return None

    def show_mfht(self, stack=False):
        for i, inputs in enumerate(self.input_iterator):
            market = inputs[0]
            start = inputs[1]
            duration = inputs[2]
            stabilvol = self.stabilvols[i]
            stabilvol_binned = self.analyst.get_average_stabilvol(stabilvol, nbins=self.inputs['nbins'])
            stabilvol_binned['Market'] = market
            stabilvol_binned['Start date'] = start
            stabilvol_binned['Window length'] = duration
            self.stabilvols_binned.append(stabilvol_binned)
            if not self.inputs['stack']:
                # Create plot and show it
                ax = self.analyst.plot_mfht(
                    stabilvol_binned,
                    title=f"{market} MFHT from {start} for {duration} years",
                    edit=True,
                )
                plt.show()
        if self.inputs['stack']:
            plot_title = f"Markets: {self.inputs['markets']} / " \
                         f"Start dates: {self.inputs['start_date']} / " \
                         f"Duration: {self.inputs['duration']}"
            ax = self.analyst.plot_mfht(
                *self.stabilvols_binned,
                # title=plot_title,
                edit=True,
            )
            plt.show()
        # Enable button to save serie
        self.btn_save['state'] = tk.NORMAL
        return None

    def save_series(self):
        for i, inputs in enumerate(self.input_iterator):
            market = inputs[0]
            start_date = inputs[1]
            duration = inputs[2]
            start = str(self.inputs["start_level"]).replace('.', 'p').replace('-', 'n')
            end = str(self.inputs["end_level"]).replace('.', 'p').replace('-', 'n')
            nbins = self.inputs["nbins"]
            fht_filename = f"{market}_{start_date}_{duration}_{start}_{end}"
            mfht_filename = f"{market}_{start_date}_{duration}_{start}_{end}_{nbins}b"
            try:
                self.analyst.save_fht(self.stabilvols[i], filename=fht_filename)
                self.analyst.save_mfht(self.stabilvols_binned[i], filename=mfht_filename)
            except FileNotFoundError as e:
                logging.warning(f"Error saving the file: {e}")
            else:
                logging.info("File saved succesfully in data/preprocessed folder")
        return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.root = ROOT

        # Style
        style = ttk.Style(self)
        style.theme_use("xpnative")

        self.title("StabilVol")
        self.geometry('800x800')
        self.resizable(True, True)

        self.label = ttk.Label(
            self,
            text="Stabilizing Effects of Volatility in Financial Markets",
            font=FONTS['h2']
        )
        self.label.grid(column=0, row=0, sticky=tk.EW, pady=15, padx=5)


if __name__ == "__main__":
    app = App()
    interface = StabilvolFrame(app)
    app.mainloop()
