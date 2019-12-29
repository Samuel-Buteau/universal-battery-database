class Print:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


    def colour(c, s):
        print(c + str(s) + Colour.END)


    def bold(c, s):
        cprint(c + Colour.bold, s)


    def type(x):
        bprint(type(x))

