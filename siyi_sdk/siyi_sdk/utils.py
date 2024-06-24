def toHex(intval, nbits):
    """
    Converts an integer to hexdecimal.
    Useful for negative integers where hex() doesn't work as expected

    Params
    --
    intaval: [int] Integer number
    nbits: [int] Number of bits

    Returns
    --
    String of the hexdecimal value
    """
    h = format((intval + (1 << nbits)) % (1 << nbits),'x')
    # if len(h)==1:
    #     h="0"+h
    while len(h)<nbits//4:
        h="0"+h
    return h

def toInt(hexval):
    """
    Converts hexidecimal value to an integer number, which can be negative
    Ref: https://www.delftstack.com/howto/python/python-hex-to-int/

    Params
    --
    hexval: [string] String of the hex value
    """
    bits = 16
    val = int(hexval, bits)
    if val & (1 << (bits-1)):
        val -= 1 << bits
    return val

def mapping(x : float, in_min : float, in_max : float, out_min : float, out_max : float) -> float:
    """
    Maps a value from one range to another range
    Ref: https://www.arduino.cc/reference/en/language/functions/math/map/

    Params
    --
    x: [float] Value to map
    in_min: [float] Lower bound of the input range
    in_max: [float] Upper bound of the input range
    out_min: [float] Lower bound of the output range
    out_max: [float] Upper bound of the output range

    Returns
    --
    Mapped value
    """
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min