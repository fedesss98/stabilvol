from tkinter import ttk

FONTS = {
    'p': ('Helvetica', 12, 'normal'),
    'h1': ('Helvetica', 24, 'bold'),
    'h2': ('Helvetica', 18, 'bold'),
    'h3': ('Helvetica', 16, 'normal'),
    'h4': ('Helvetica', 12, 'bold'),
    'p-help': ('Helvetica', 9, 'normal')
}

STYLES = {
    'border': {'highlightthickness': .5,
               'highlightcolor': '#0C59F2',
               'highlightbackground': '#EDEDED'
               }
}

COLORS = {
    'light': '#FFFFFF',
    'mid': '#EDEDED',
    'dark': '#888888',
}


class StabilvolStyle(ttk.Style):
    """
    Custom Styles for Tkinter
    """
    def __init__(self):
        super(StabilvolStyle, self).__init__()
        self.theme_use('xpnative')

        # Light Frame
        self.configure('TFrame', background=COLORS['light'])
        self.configure('TLabelframe', background=COLORS['light'])
        self.configure('H1.TLabelframe.Label', font=FONTS['h3'], background=COLORS['light'])
        self.configure('TLabel', background=COLORS['light'])
        self.configure('TEntry', background=COLORS['light'])
        self.configure('TRadiobutton', background=COLORS['light'])
        # Dark Frame
        self.configure('DarkFrame.TFrame', background=COLORS['mid'], highlightbackground=COLORS['dark'],
                       highlightcolor=COLORS['dark'], highlightthickness=2, borderwidth=2)
        self.configure('DarkFrame.TLabel', background=COLORS['mid'])

    @property
    def h1(self):
        return FONTS['h1']

    @property
    def h2(self):
        return FONTS['h2']

    @property
    def h3(self):
        return FONTS['h3']

    @property
    def p(self):
        return FONTS['p']