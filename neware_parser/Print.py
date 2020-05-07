class Print:
    PINK = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @staticmethod
    def red(s):
        Print.colour(Print.RED, s)

    @staticmethod
    def yellow(s):
        Print.colour(Print.YELLOW, s)

    @staticmethod
    def blue(s):
        Print.colour(Print.BLUE, s)

    @staticmethod
    def green(s):
        Print.colour(Print.GREEN, s)

    @staticmethod
    def pink(s):
        Print.colour(Print.PINK, s)

    @staticmethod
    def colour(c, s):
        print(c + str(s) + Print.END)

    @staticmethod
    def bold(c, s):
        Print.colour(c + Print.BOLD, s)

    @staticmethod
    def type(x):
        Print.bold(type(x))
