"""
Graphic User Interface
"""
try:
    from .utility.definitions import ROOT
    from .utility.styles import FONTS
    from .data.count_stocks import main as count_stocks
except ImportError:
    from stabilvol.utility.definitions import ROOT
    from stabilvol.utility.styles import FONTS
    from stabilvol.data.count_stocks import main as count_stocks

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
MARKETS = [file.stem for file in (ROOT/'data/raw').glob('*.pickle')]


class MarketSelectionFrame(ttk.Frame):
    """
    Market selection Frame.
    You can choose one or more market from MARKETS to analyze.
    """
    def __init__(self, container, col, row, rowspan):
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
            listvariable=self._markets,
            height=len(MARKETS),
            selectmode=tk.MULTIPLE,
            exportselection=0,
            font=FONTS['p'],
        )
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
            self,
            text="Select the lengths of the Windows, separated by a comma.",
            font=FONTS['p-help'],
        )
        self.window_lengths = tk.Text(self, height=1, width=20)
        self.window_lengths.insert('1.0', "10")
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.window_lengths.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)

    @property
    def windows(self):
        return list(self.window_lengths.get(1.0, 'end-1c').split(', '))


class StartTextFrame(ttk.Frame):
    """
    Windows selection Frame.
    You can enter the lengths of the windows to use.
    """
    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Start", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self,
            text="Insert the starting date for the windows (YYYY-MM-DD)",
            font=FONTS['p-help'],
        )
        self.start_string = tk.Text(self, height=1, width=20)
        self.start_string.insert('1.0', "1980-01-01")
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, sticky=tk.W)
        self.start_string.grid(column=0, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)

    @property
    def start_date(self):
        return self.start_string.get(1.0, 'end-1c')


class CriterionDropdownFrame(ttk.Frame):
    """
    Criterion selection Frame.
    You can choose one criterion between 'percentage' and 'threshold' and add a value.
    """
    def __init__(self, container, col, row):
        super().__init__(container)

        self.frame_label = ttk.Label(self, text="Criterion", font=FONTS['h4'])
        self.frame_label_help = ttk.Label(
            self,
            text="Choose one criterion and the threshold\n(Float for Percentage and int number of days for StartEnd)",
            font=FONTS['p-help'],
        )
        self._criterion = tk.StringVar()
        self.criterion_selection = ttk.Combobox(self,
                                                state='readonly',
                                                values=["Percentage", "StartEnd"],
                                                textvariable=self._criterion)
        self._criterion.set("Percentage")
        self.value_input = tk.Text(self, height=1, width=4)
        self.value_input.insert('1.0', "0.8")
        # Place Widgets
        self.frame_label.grid(column=0, row=0, columnspan=2, sticky=tk.W)
        self.frame_label_help.grid(column=0, row=1, columnspan=2, sticky=tk.W)
        self.criterion_selection.grid(column=0, row=2, sticky=tk.W)
        self.value_input.grid(column=1, row=2, sticky=tk.W)
        # Place Frame.
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.W)

    @property
    def criterion(self):
        return self._criterion.get()

    @property
    def criterion_value(self):
        return self.value_input.get(1.0, 'end-1c')


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
        self._save_choice = tk.StringVar()
        self.yes_choice = ttk.Radiobutton(self, text="Yes", value="True", variable=self._save_choice)
        self.no_choice = ttk.Radiobutton(self, text="No", value="False", variable=self._save_choice)
        self._save_choice.set("False")
        # Place Widgets
        self.frame_label.grid(column=0, row=0, sticky=tk.W)
        self.yes_choice.grid(column=0, row=1, sticky=tk.EW)
        self.no_choice.grid(column=0, row=2, sticky=tk.EW)
        # Place Frame
        self.grid(column=col, row=row, padx=5, pady=5, sticky=tk.EW)

    @property
    def choice(self):
        return self._save_choice.get()


class StocksCountingFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        # Initialize inputs
        self.markets = []
        self.windows = [10]
        self.start_date = '1980-01-01'
        self.criterion = 'percentage'
        self.criterion_value = 0.8
        self.slide = False
        self.end_date = None
        self.save = False

        # field options
        options = {'padx': 5, 'pady': 5}
        self.config(borderwidth=3, relief='groove')

        # label
        self.label = ttk.Label(self, text='Stocks Counting in Window', font=FONTS['h3'])
        self.label.grid(column=0, row=0, columnspan=2, **options)

        self.marketselection = MarketSelectionFrame(self, col=0, row=1, rowspan=2)
        self.windowsinputs = WindowsTextFrame(self, col=0, row=3)
        self.startinput = StartTextFrame(self, col=0, row=4)
        self.criterionselection = CriterionDropdownFrame(self, col=1, row=1)
        self.slidechoice = SlideChoiceFrame(self, col=1, row=2)
        self.savechoice = SaveChoiceFrame(self, col=1, row=3)

        # Button
        self.button = ttk.Button(self, text='Count!', width=10, command=self.start_counting)
        self.button.grid(column=1, row=4, sticky=tk.EW, padx=5, pady=5)

        self.inputs_frame = ttk.LabelFrame(self, text="Inputs", width=600)
        self.inputs_label = ttk.Label(self.inputs_frame, text="Inputs will be displayed here")
        self.inputs_frame.grid(column=0, row=5, columnspan=2, sticky=tk.NSEW, **options)
        self.inputs_label.pack(fill='x')

        # add padding to the frame and show it
        self.grid(column=0, row=1, padx=20, pady=10, sticky=tk.NSEW)

    def retrieve_inputs(self):
        self.markets = self.marketselection.markets
        self.windows = self.windowsinputs.windows
        self.start_date = self.startinput.start_date
        self.criterion = self.criterionselection.criterion.lower()
        self.criterion_value = self.criterionselection.criterion_value
        self.slide = self.slidechoice.choice == "True"
        self.end_date = self.slidechoice.end_date
        self.save = self.savechoice.choice == "True"
        input_text = f"""
            Markets: {', '.join(self.markets)};
            Windows: {', '.join(self.windows)};
            Start Date: {self.start_date};
            Criterion: {self.criterion} ({self.criterion_value})
            - Slide {self.slide} (end date: {self.end_date})
            - Save {self.save}
        """
        print(
            f"{self.markets} : {type(self.markets)}",
            f"{self.windows} : {type(self.windows)}",
            f"{self.start_date} : {type(self.start_date)}",
            f"{self.criterion} : {type(self.criterion)}",
            f"{self.criterion_value} : {type(self.criterion_value)}",
            sep="\n"
        )
        self.inputs_label.config(text=input_text)
        if len(self.markets) == 0:
            messagebox.showerror(title="No market selected!",
                                 message="Select at least one market.")
            raise ValueError("No market selected")

    def start_counting(self):
        self.retrieve_inputs()
        print("*** OUTPUT ***")
        count_stocks(**self.inputs)

    @property
    def inputs(self):
        return {
            'markets': self.markets,
            'window_lengths': self.windows,
            'start': self.start_date,
            'criterion': {self.criterion: self.criterion_value},
            'sliding': self.slide,
            'end': self.end_date,
            'save': self.save,
        }


class OutputFrame(ttk.LabelFrame):
    def __init__(self, container, inputs):
        super().__init__(container, text="Output", height=400)
        # field options

        self.output_menu = ttk.Notebook(self)
        self.text_frame = tk.Label(self.output_menu, text="Output will be displayed here.")

        self.output_menu.pack()
        self.text_frame.pack(fill='x')

        # add padding to the frame and show it
        self.grid(column=0, row=2, padx=20, pady=10, sticky=tk.NSEW)
        self.grid_propagate(0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()

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
    interface = StocksCountingFrame(app)
    OutputFrame(app, interface.inputs)
    app.mainloop()
