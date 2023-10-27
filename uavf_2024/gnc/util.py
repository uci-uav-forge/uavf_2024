def read_gps(fname):
    with open(fname) as f:
        return [tuple(map(float, line.split(','))) for line in f]