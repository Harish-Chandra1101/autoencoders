import io
import numpy as np
from numpy import genfromtxt


def read_csv(request, name):
    file_obj = io.StringIO(request.FILES[name].read().decode('UTF-8'))
    np_data = genfromtxt(file_obj, delimiter=',', skip_header=1, names=True)
    return np_data


def remove_nan(x):
    x = x[~np.isnan(x)]
    return x
