# `D:\src\scipysrc\matplotlib\lib\matplotlib\pylab.py`

```
"""
`pylab` is a historic interface and its use is strongly discouraged. The equivalent
replacement is `matplotlib.pyplot`.  See :ref:`api_interfaces` for a full overview
of Matplotlib interfaces.

`pylab` was designed to support a MATLAB-like way of working with all plotting related
functions directly available in the global namespace. This was achieved through a
wildcard import (``from pylab import *``).

.. warning::
   The use of `pylab` is discouraged for the following reasons:

   ``from pylab import *`` imports all the functions from `matplotlib.pyplot`, `numpy`,
   `numpy.fft`, `numpy.linalg`, and `numpy.random`, and some additional functions into
   the global namespace.

   Such a pattern is considered bad practice in modern python, as it clutters the global
   namespace. Even more severely, in the case of `pylab`, this will overwrite some
   builtin functions (e.g. the builtin `sum` will be replaced by `numpy.sum`), which
   can lead to unexpected behavior.

"""

# Importing utility functions from matplotlib
from matplotlib.cbook import flatten, silent_list

# Importing matplotlib library under the name 'mpl'
import matplotlib as mpl

# Importing various date-related utilities from matplotlib
from matplotlib.dates import (
    date2num, num2date, datestr2num, drange, DateFormatter, DateLocator,
    RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator,
    HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR,
    SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY,
    relativedelta)

# Importing specific data processing functions from matplotlib.mlab
# Note: Additional cleanup is recommended for imports from mlab
from matplotlib.mlab import (
    detrend, detrend_linear, detrend_mean, detrend_none, window_hanning,
    window_none)

# Importing pyplot module from matplotlib and renaming it as 'plt'
from matplotlib import cbook, mlab, pyplot as plt
from matplotlib.pyplot import *

# Importing all functions and objects from numpy
from numpy import *

# Importing FFT-related functions from numpy.fft
from numpy.fft import *

# Importing random number generation functions from numpy.random
from numpy.random import *

# Importing linear algebra functions from numpy.linalg
from numpy.linalg import *

# Importing numpy under the name 'np' and numpy masked array functionality as 'ma'
import numpy as np
import numpy.ma as ma

# Importing standard datetime module from Python standard library
import datetime

# Avoiding name conflicts with built-in functions and modules
# by explicitly importing necessary functions from builtins
bytes = __import__("builtins").bytes
abs = __import__("builtins").abs
bool = __import__("builtins").bool
max = __import__("builtins").max
min = __import__("builtins").min
pow = __import__("builtins").pow
round = __import__("builtins").round
```