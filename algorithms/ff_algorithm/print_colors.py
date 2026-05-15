# -*- coding: utf-8 -*-

class Colors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    # B means BRIGHT
    B_BLACK = '\033[90m' # Bright black
    B_RED = '\033[91m'
    B_GREEN = '\033[92m'
    B_YELLOW = '\033[93m'
    B_BLUE = '\033[94m'
    B_MAGENTA = '\033[95m'
    B_CYAN = '\033[96m'
    B_WHITE = '\033[97m'
    B_END = '\033[0m'

if __name__ == '__main__':
    print(Colors.B_BLACK + 'TEST' + Colors.RESET)
    print(Colors.B_RED + 'TEST' + Colors.RESET)
    print(Colors.B_GREEN + 'TEST' + Colors.RESET)
    print(Colors.B_YELLOW + 'TEST' + Colors.RESET)
    print(Colors.B_BLUE + 'TEST' + Colors.RESET)
    print(Colors.B_MAGENTA + 'TEST' + Colors.RESET)
    print(Colors.B_CYAN + 'TE' + Colors.BLUE + 'ST' + Colors.RESET)
    print(Colors.B_WHITE + 'TE' + Colors.BLUE + 'ST' + Colors.RESET)

    print(Colors.BLACK + 'TEST' + Colors.RESET)
    print(Colors.RED + 'TEST' + Colors.RESET)
    print(Colors.GREEN + 'TEST' + Colors.RESET)
    print(Colors.YELLOW + 'TEST' + Colors.RESET)
    print(Colors.BLUE + 'TEST' + Colors.RESET)
    print(Colors.MAGENTA + 'TEST' + Colors.RESET)
    print(Colors.CYAN + 'TE' + Colors.BLUE + 'ST' + Colors.RESET)
    print(Colors.WHITE + 'TE' + Colors.BLUE + 'ST' + Colors.RESET)
    print(Colors.UNDERLINE + 'TE' + Colors.BLUE + 'ST' + Colors.RESET)