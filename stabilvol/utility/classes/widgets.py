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
    from stabilvol.utility.styles import FONTS, STYLES
    from stabilvol.utility.classes.data_extraction import DataExtractor
    from stabilvol.utility.classes.stability_analysis import StabilVolter
    from stabilvol.analyze.count_fht import main as count_fht
except ModuleNotFoundError as e:
    logging.warning(f"Error in count_fht imports: {e}")
    from utility.definitions import ROOT
    from utility.styles import FONTS, STYLES
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
        super().__init__(container, style='DarkFrame.TFrame')
        self.controller = container.controller

        self.frame_label = ttk.Label(self, text="Markets", font=FONTS['h4'], style='DarkFrame.TLabel')
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], width=32, style='DarkFrame.TLabel',
            text="Select one or more Markets from the list",
        )

        self._markets = tk.Variable(value=list(MARKETS))
        self.markets_list = tk.Listbox(
            self,
            listvariable=self.controller.available_markets,
            height=len(MARKETS),
            selectmode=tk.EXTENDED,
            exportselection=0,
            font=self.controller.style.p,
        )
        # Colorize alternating lines of the listbox
        for i in range(0, len(MARKETS), 2):
            self.markets_list.itemconfigure(i, background='#f0f0ff')

        # Binding market data extraction to left mouse double click
        self.markets_list.bind('<Double-Button-1>', self.controller.select_stocks)
        # Binding market info to market click
        self.markets_list.bind('<<ListboxSelect>>', self.controller.describe_selected_market)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.markets_list.grid(column=0, row=2, sticky=tk.EW)
        # Place Frame.
        self.grid(column=col, row=row, rowspan=rowspan, padx=(20, 10), pady=(30, 10), sticky=tk.W)

    @property
    def markets(self):
        return [self.markets_list.get(i) for i in self.markets_list.curselection()]


class WindowsTextFrame(ttk.Frame):
    """
    Windows selection Frame.
    You can enter the lengths of the windows to use.
    """

    def __init__(self, controller, col, row):
        super().__init__(controller)

        self.frame_label = ttk.Label(self, text="Windows", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Select the lengths of the Windows, separated by a comma.",
        )
        self.window_lengths = tk.Entry(self, textvariable=controller.window_length, **STYLES['border'])
        # self.window_lengths.insert('1.0', "10")
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.window_lengths.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=20, pady=20, sticky=tk.W)


class StartTextFrame(ttk.Frame):
    """
    Frame to select the starting date for the Window.
    """

    def __init__(self, controller, col, row):
        super().__init__(controller)

        self.frame_label = ttk.Label(self, text="Start Date", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Insert the starting date for the windows (YYYY-MM-DD)",
        )
        self.start_string = tk.Entry(self, textvariable=controller.start_date, width=15, **STYLES['border'])
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.start_string.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=20, pady=20, sticky=tk.W)

    @property
    def start_date(self):
        return self.start_string.get()


class EndTextFrame(ttk.Frame):
    def __init__(self, controller, col, row):
        super().__init__(controller)

        self.frame_label = ttk.Label(self, text="End Date", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self,
            text="Insert the ending date for the windows (YYYY-MM-DD or YYYY)",
            font=FONTS['p-help'],
        )
        self.start_string = tk.Entry(self, textvariable=controller.end_date, **STYLES['border'])
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
        self.controller = container.controller

        self.frame_label = ttk.Label(self, text="Criterion", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=280,
            text="Choose one selection criterion for returns "
                 "(Float for Percentage and int number of days for StartEnd)",
        )
        # Help with the criterion value
        self._value_label_help = tk.StringVar(self, value='Days:')
        self.value_label_help = ttk.Label(
            self, font=FONTS['p-help'], textvariable=self._value_label_help
        )
        self.criterion_selection = ttk.Combobox(self,
                                                state='readonly',
                                                values=["Percentage", "StartEnd"],
                                                textvariable=self.controller.criterion)
        self.value_input = tk.Entry(self, textvariable=self.controller.criterion_value, width=10, **STYLES['border'])
        # Bind change in combox with change in help for criterion value
        self.criterion_selection.bind('<<ComboboxSelected>>', self.update_help)
        # Place Widgets
        self.frame_label.grid(column=0, row=0, columnspan=2, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, columnspan=2, sticky=tk.W)
        self.criterion_selection.grid(column=0, row=2, columnspan=2, sticky=tk.NW)
        self.value_label_help.grid(column=0, row=3, sticky=tk.NW, pady=4)
        self.value_input.grid(column=0, row=4, sticky=tk.NW, pady=4)
        # Place Frame
        self.grid(column=col, row=row, padx=20, pady=(15, 12), sticky=tk.NSEW)

    def update_help(self, event):
        if self.controller.criterion.get() == "StartEnd":
            self._value_label_help.set("Days:")
        elif self.controller.criterion.get() == "Percentage":
            self._value_label_help.set("Fraction:")


class SlideChoiceFrame(ttk.Frame):
    """
    Windows selection Frame.
    You can enter the lengths of the windows to use.
    """

    def __init__(self, controller, col, row):
        super().__init__(controller)

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

    def __init__(self, controller, col, row):
        super().__init__(controller)

        self.frame_label = ttk.Label(self, text="Save", font=FONTS['h4'])
        self._save_choice = tk.BooleanVar(self, value=False)
        self.yes_choice = ttk.Radiobutton(self, text="Yes", value=True, variable=controller.save)
        self.no_choice = ttk.Radiobutton(self, text="No", value=False, variable=controller.save)
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

    def __init__(self, controller, col, row, variable):
        super().__init__(controller)
        self.variable = variable if variable else tk.StringVar()
        self.frame_label = ttk.Label(self, text="Start Level: ", font=FONTS['p-help'])
        self.start_string = tk.Entry(self, textvariable=variable, width=8, **STYLES['border'])
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.start_string.grid(column=1, row=0, sticky=tk.W, padx=2)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class EndLevelTextFrame(ttk.Frame):
    """
    Frame to select the ending threshold for FHT counting.
    """

    def __init__(self, controller, col, row, variable=None):
        super().__init__(controller)
        self.variable = variable if variable else tk.StringVar()
        self.frame_label = ttk.Label(self, text="End Level:  ", font=FONTS['p-help'])
        self.end_string = tk.Entry(self, textvariable=self.variable, width=8, **STYLES['border'])
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.end_string.grid(column=1, row=0, sticky=tk.W, padx=2)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)


class ThresholdFrame(ttk.Frame):
    """ Unified container for StartLevel and EndLevel entries """

    def __init__(self, container, col, row):
        super().__init__(container)
        self.controller = container.controller

        self.frame_label = ttk.Label(self, text="Thresholds", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Insert threshold levels for FHT counting",
        )
        # Place widgets
        self.frame_label.grid(column=0, row=0, columnspan=2, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, columnspan=2, sticky=tk.W)
        self.start = StartLevelTextFrame(self, col=0, row=2, variable=self.controller.start_level)
        self.end = EndLevelTextFrame(self, col=0, row=3, variable=self.controller.end_level)
        # Place Frame
        self.grid(column=col, row=row, padx=20, pady=12, sticky=tk.NSEW)


class NbinsFrame(ttk.Frame):
    def __init__(self, container, col, row):
        super().__init__(container)
        self.controller = container.controller

        self.frame_label = ttk.Label(self, text="Number of bins", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=250,
            text="Insert the number of bins in which to average the FHT",
        )
        self.num_bins = tk.Entry(self, textvariable=self.controller.nbins, **STYLES['border'])
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.num_bins.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=20, pady=12, sticky=tk.NSEW)


class StackChoiceFrame(ttk.Frame):
    """
    Choose wether to stack graph on one column or plot them separately.
    """

    def __init__(self, container, col, row):
        super().__init__(container)
        self.controller = container.controller

        # Insert separator before widget
        ttk.Separator(self, orient=tk.HORIZONTAL).grid(column=0, row=0, ipadx=150, pady=10, sticky=tk.EW)

        self.frame_label = ttk.Label(self, text="Stack plots", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], wraplength=300,
            text="Select the variable on which to stack the plots:"
        )
        stack_options = ["Nothing", "Market", "Start date", "Window length"]
        self.stack_select = ttk.Combobox(self,
                                         state="readonly",
                                         values=stack_options,
                                         textvariable=self.controller.stack)
        # Place Widgets
        self.frame_label.grid(column=0, row=1, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=2, sticky=tk.W)
        self.stack_select.grid(column=0, row=3, sticky=tk.EW)
        # Place Frame
        self.grid(column=col, row=row, padx=20, pady=12, sticky=tk.EW)


class ReturnsSelection(ttk.Frame):
    """
    Returns selection Frame.
    Display stocks in selected market.
    """

    def __init__(self, container, col=1, row=1, rowspan=1):
        super().__init__(container, style='DarkFrame.TFrame')
        self.controller = container.controller

        self.frame_label = ttk.Label(self, text="Stocks", font=FONTS['h4'], style='DarkFrame.TLabel')
        self.frame_label_help = ttk.Label(
            self, font=FONTS['p-help'], style='DarkFrame.TLabel', width=32,
            text="Stocks in selected market.\nDouble click to show returns",
        )
        self.stocks_list = tk.Listbox(
            self,
            listvariable=self.controller.stocks,
            height=15,
            selectmode=tk.BROWSE,
            exportselection=0,
            font=FONTS['p'],
        )
        # Binding double click left mouse
        self.stocks_list.bind('<Double-Button-1>', self.controller.show_returns)
        # Colorize alternating lines of the listbox
        for i in range(0, len(self.controller.stocks.get()), 2):
            self.stocks_list.itemconfigure(i, background='#f0f0ff')

        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.stocks_list.grid(column=0, row=2, sticky=tk.EW)
        # Place Frame.
        self.grid(column=col, row=row, rowspan=rowspan, padx=20, pady=5, sticky=tk.W)

    @property
    def selected_market(self):
        if len(self.controller.marketselection.markets) > 0:
            return self.controller.marketselection.markets[0]
        else:
            return None

    @property
    def selected_stock(self):
        return self.stocks_list.curselection()[0]


class SettingsWindow(tk.Toplevel):
    def __init__(self, controller):
        super(SettingsWindow, self).__init__(controller)
        self.controller = controller
        self.configure(background='#FFFFFF')
        self.title("Settings")
        self.geometry("500x1000")
        self.resizable(False, False)
        # Configure the grid
        self.grid_columnconfigure(0, weight=1)
        for i in range(8):
            self.grid_rowconfigure(i, weight=1)
        # Labels
        self.label_title = ttk.Label(self, font=self.controller.style.h2,
                                     text="Settings"
                                     )
        self.label = ttk.Label(self, font=self.controller.style.p,
                               text="Manage basic settings to count FHT"
                               )
        self.label_title.grid(column=0, row=0)
        self.label.grid(column=0, row=1)
        ttk.Separator(self, orient=tk.HORIZONTAL).grid(column=0, row=2, pady=4, sticky=tk.EW)
        # WIDGETS
        self.criterioniputs = CriterionDropdownFrame(self, col=0, row=3)
        self.thresholdsinputs = ThresholdFrame(self, col=0, row=4)
        self.nbinsinput = NbinsFrame(self, col=0, row=5)
        self.stackinput = StackChoiceFrame(self, col=0, row=6)
        # Add button to close and save
        self.btn_save = ttk.Button(self, text="Save", command=self.destroy)
        self.btn_save.grid(column=0, row=7, padx=20, pady=(20, 25), ipadx=10, ipady=10)


class MenuBar(tk.Menu):
    def __init__(self, controller):
        super(MenuBar, self).__init__(controller)

        self.controller = controller
        self.file_menu = tk.Menu(self, tearoff=0)
        self.help_menu = tk.Menu(self, tearoff=0)
        # File Menu options
        self.file_menu.add_command(label="Reset")
        self.file_menu.add_command(label="Open")
        self.file_menu.add_command(label="Settings", command=self.controller.open_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.controller.destroy)
        # Add the File Menu to the Menubar
        self.add_cascade(
            label="File",
            menu=self.file_menu,
            underline=0,
        )
        # Help Menu options
        self.help_menu.add_command(label="About")
        self.help_menu.add_command(label="Help")
        # Add the File Menu to the Menubar
        self.add_cascade(
            label="Help",
            menu=self.help_menu
        )


class LeftBar(ttk.Frame):
    def __init__(self, controller):
        super(LeftBar, self).__init__(controller, style='DarkFrame.TFrame')
        self.controller = controller

        # Widgets
        self.marketselection = MarketSelectionFrame(self, col=0, row=0)
        self.returnselection = ReturnsSelection(self, col=0, row=2)
        # Buttons
        # Select market
        self.btn_select = ttk.Button(self, text='Open returns file')
        # Show selected stocks
        self.btn_show = ttk.Button(self, text='Show selected stocks in Market', command=self.controller.select_stocks)
        self.btn_show.grid(column=0, row=1, sticky=tk.EW, padx=20, pady=5)

        self.grid(column=0, row=0, rowspan=4, ipadx=20, ipady=20, sticky=tk.NSEW)


class StabilvolFrame(ttk.LabelFrame):
    def __init__(self, controller):
        super().__init__(controller, text='FHT Counting', style='H1.TLabelframe')

        self.stabilvols_binned = []
        self.controller = controller
        # Variables
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
        self.accountant = DataExtractor(start_date=self.start_date.get(),
                                        end_date=self.end_date.get(),)
        self.datas = []
        self.analyst = StabilVolter()
        self.stabilvols = []
        self.mfhts = []

        # Fields options
        options = {'padx': 5, 'pady': 5}

        # Widgets
        # Col 2
        self.startinput = StartTextFrame(self, col=2, row=1)
        self.windowsinputs = WindowsTextFrame(self, col=2, row=2)

        # add padding to the frame and show it
        self.grid(column=1, row=1, padx=10, pady=10, ipadx=5, ipady=10, sticky=tk.NSEW)


class ResultsFrame(ttk.LabelFrame):
    """
    Show results of the counting.
    - Logger ID of the run
    - Max value of MFHT
    - 1st and 2nd baricenter of MFHT
    """
    def __init__(self, controller):
        super(ResultsFrame, self).__init__(controller, text='Count Results', style='H1.TLabelframe')

        self.labels = ttk.Label(self, text="Results will be displayed here.")
        self.label_log = ttk.Label(self, text="Log ID: ")

        # Place Labels
        self.labels.grid(column=0, row=0)
        self.label_log.grid(column=0, row=1)
        # Place Frame
        self.grid(column=1, row=2, padx=10, pady=5, sticky=tk.NSEW)


class ButtonFrame(ttk.Frame):
    def __init__(self, controller):
        super(ButtonFrame, self).__init__(controller)
        self.controller = controller
        self.btn_count = ttk.Button(self, text="COUNT", width=14, command=controller.start_counting_cycles)
        self.btn_plot_fht = ttk.Button(self, text="PLOT FHT", width=14, command=self.plot_fht)
        self.btn_plot_pdf = ttk.Button(self, text="PLOT PDF", width=14, command=self.plot_pdf)
        self.btn_plot_mfht = ttk.Button(self, text="PLOT MFHT", width=14, command=self.plot_mfht)
        self.btn_save = ttk.Button(self, text="SAVE", width=12, command=controller.save_analysis)
        # self.btn_show = ttk.Button(self, text="VISUALIZE", width=15, )
        # Place Widgets
        self.btn_count.grid(row=0, column=0, padx=8, ipadx=5, ipady=10, sticky=tk.E)
        self.btn_plot_fht.grid(row=0, column=1, padx=8, ipadx=5, ipady=10, sticky=tk.E)
        self.btn_plot_pdf.grid(row=0, column=2, padx=8, ipadx=5, ipady=10, sticky=tk.E)
        self.btn_plot_mfht.grid(row=0, column=3, padx=8, ipadx=5, ipady=10, sticky=tk.E)
        self.btn_save.grid(row=0, column=4, padx=8, ipadx=5, ipady=10, sticky=tk.E)
        # self.btn_show.grid(row=02, column=4, padx=8, ipadx=5, ipady=10, sticky=tk.E)
        # Place Frame
        self.grid(column=1, row=3, padx=10, pady=20, sticky=tk.E)

    def plot_fht(self):
        self.controller.plot_selected_data(self.controller.fhts)

    def plot_pdf(self):
        self.controller.plot_selected_data(self.controller.pdfs, data_type='pdf')

    def plot_mfht(self):
        self.controller.plot_selected_data(self.controller.mfhts)


class StatusFrame(ttk.Frame):
    """
    Show status of the app:
    - number of stocks selected
    - std of the market
    - counted/not counted fht
    """

    def __init__(self, controller):
        super().__init__(controller, relief=tk.RIDGE)
        self.controller = controller
        self.controller.done_counting.trace_add('write', self.update_count)

        self.label_market = ttk.Label(self, text='Selected Market: - ', )
        self.label_stocks = ttk.Label(self, text='Stock included in period: - ', )
        self.label_count = ttk.Label(self, text=f'FHT: Not counted', )
        # Place Widgets
        self.label_market.grid(column=0, row=0, padx=15, pady=(10, 0), sticky=tk.W)
        self.label_stocks.grid(column=1, row=0, padx=15, pady=(10, 0), sticky=tk.W)
        self.label_count.grid(column=2, row=0, padx=15, pady=(10, 0), sticky=tk.E)
        # Place Frame
        self.grid(column=0, row=4, columnspan=2, ipadx=5, ipady=5, sticky=tk.NSEW)
        # Create grid
        for c in range(5):
            self.columnconfigure(c, weight=1)

    def update_count(self, *args):
        count_status = 'Counted' if self.controller.done_counting.get() else 'Not counted'
        self.label_count['text'] = f'FHT: {count_status}'
        return None


class LoadingWindow(tk.Toplevel):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.configure(background='#FFFFFF')
        self.title("Counting FHT...")
        self.geometry("650x450+600+600")
        self.resizable(False, False)
        self.label = ttk.Label(self, text="Loading...")
        self.label.pack(padx=10, pady=10)
        self.progressbar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progressbar.pack(padx=10, pady=10)
        self.progressbar.start(5)

    def on_closing(self):
        self.controller.stop_counting()
        self.destroy()




class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.root = ROOT

        self.done_counting = tk.BooleanVar(self, value=False)

        # Style
        self.configure(background='#FFFFFF')
        style = ttk.Style(self)
        style.theme_use("xpnative")
        style.configure('TFrame', background='#FFFFFF')
        style.configure('TLabelframe', background='#FFFFFF')
        style.configure('H1.TLabelframe.Label', font='Helvetica 16', background='#FFFFFF')
        style.configure('TLabel', background='#FFFFFF')
        style.configure('TEntry', background='#FFFFFF')
        style.configure('TRadiobutton', background='#FFFFFF')
        style.configure('DarkFrame.TFrame', background='#EDEDED', highlightbackground='#888888',
                        highlightcolor='#888888', highlightthickness=2, borderwidth=2)
        style.configure('DarkFrame.TLabel', background='#EDEDED')

        self.title("StabilVol")
        self.geometry('1080x700')
        self.resizable(True, True)
        self.minsize(930, 630)

        self.label = ttk.Label(
            self,
            text="Stabilizing Effects of Volatility in Financial Markets",
            font=FONTS['h2']
        )
        self.label.grid(column=1, row=0, sticky=tk.W, pady=15, padx=5)

        # Configure the grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=4)
        self.rowconfigure(0, weight=2)
        self.rowconfigure(1, weight=5)
        self.rowconfigure(2, weight=2)
        self.rowconfigure(2, weight=1)


if __name__ == "__main__":
    app = App()
    menubar = MenuBar(app)
    app.config(menu=menubar)
    marketframe = LeftBar(app)
    interface = StabilvolFrame(app)
    buttons = ButtonFrame(app)
    statusbar = StatusFrame(app)
    app.mainloop()
