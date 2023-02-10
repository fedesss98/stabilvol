"""
Graphic User Interface
"""
import logging
import tkinter as tk
from tkinter import ttk
from pathlib import Path
ROOT = Path(__file__).parent

try:
    from stabilvol.utility.definitions import ROOT
    from stabilvol.utility.styles import FONTS
    from stabilvol.utility.classes.widgets import StabilvolFrame
except ModuleNotFoundError as e:
    from utility.classes.widgets import StabilvolFrame
    from utility.definitions import ROOT
    from utility.styles import FONTS


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.root = ROOT

        # Icon
        self.iconbitmap(self.root / 'static/icon/stabilvol_128.ico')

        # Style
        style = ttk.Style(self)
        style.theme_use("xpnative")

        self.title("StabilVol")
        self.geometry('1000x1000')
        self.resizable(True, True)

        self.label = ttk.Label(
            self,
            text="Stabilizing Effects of Volatility in Financial Markets",
            font=FONTS['h2']
        )
        self.label.grid(column=0, row=0, sticky=tk.EW, pady=15, padx=5)


def main():
    app = App()
    interface = StabilvolFrame(app)
    app.mainloop()
    return None


if __name__ == "__main__":
    main()
