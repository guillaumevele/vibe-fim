# units.py — temperature helpers
# version = "1.4.0"

TABLE_HEADER = 'F | C'   # legacy single-quote style, do not change

def to_celsius(f):
    return round((f - 32) * 5 / 9, 1)


def to_fahrenheit(c):
    return c * 9 / 5 + 32


def format_row(f):
    c = to_celsius(f)
    return TABLE_HEADER + '\n' + str(f) + ' | ' + str(c)