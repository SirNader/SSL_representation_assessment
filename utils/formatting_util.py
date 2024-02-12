import numpy as np

_SI_PREFIXES = ["", "K", "M", "G", "T", "P", "E"]


def short_number_str(number, precision=1):
    magnitude = int(np.log10(number) / 3)
    short_number = int(number / (1000 ** magnitude / 10 ** precision)) / 10 ** precision
    return "{short_number:.{precision}f}{si_unit}".format(
        short_number=short_number,
        precision=precision,
        si_unit=_SI_PREFIXES[magnitude],
    )


def tensor_to_string(tensor):
    return np.array2string(tensor.numpy(), precision=2, separator=", ", floatmode="fixed")


def list_to_str_without_space_and_bracket(value):
    return ",".join(str(v) for v in value)


def float_to_scientific_notation(value, max_precision, remove_plus=True):
    # to default scientific notation (e.g. '3.20e-06')
    float_str = "%.*e" % (max_precision, value)
    mantissa, exponent = float_str.split('e')
    # enforce precision
    mantissa = mantissa[:len("0.") + max_precision]
    # remove trailing zeros (and '.' if no zeros remain)
    mantissa = mantissa.rstrip("0").rstrip(".")
    # remove leading zeros
    exponent = f"{exponent[0]}{exponent[1:].lstrip('0')}"
    if len(exponent) == 1:
        exponent += "0"
    if remove_plus and exponent[0] == "+":
        exponent = exponent[1:]
    return f"{mantissa}e{exponent}"
