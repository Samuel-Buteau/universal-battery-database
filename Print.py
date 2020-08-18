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
    def colour(c, s):
        print(c + str(s) + Print.END)

    @staticmethod
    def bold(c, s):
        colour(c + Print.BOLD, s)

    @staticmethod
    def type(x):
        bold(type(x))
