# `D:\src\scipysrc\seaborn\seaborn\_stats\regression.py`

```
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from seaborn._stats.base import Stat

@dataclass
class PolyFit(Stat):
    """
    Fit a polynomial of the given order and resample data onto predicted curve.
    """
    # This is a dataclass inheriting from Stat, designed to perform polynomial fitting and data resampling.

    order: int = 2  # Default polynomial order for fitting
    gridsize: int = 100  # Number of points to interpolate over

    def _fit_predict(self, data):
        """
        Perform polynomial fitting and return interpolated data.

        Parameters:
        - data: A DataFrame containing 'x' and 'y' columns.

        Returns:
        - DataFrame with interpolated 'x' and 'y' values.
        """
        x = data["x"]
        y = data["y"]
        if x.nunique() <= self.order:
            # If the number of unique x values is less than or equal to the order of the polynomial,
            # there isn't enough data to fit the polynomial. Return empty arrays for xx and yy.
            xx = yy = []
        else:
            # Fit polynomial of specified order to data
            p = np.polyfit(x, y, self.order)
            # Generate grid of x values for interpolation
            xx = np.linspace(x.min(), x.max(), self.gridsize)
            # Compute corresponding y values on the fitted polynomial
            yy = np.polyval(p, xx)

        return pd.DataFrame(dict(x=xx, y=yy))

    # TODO: Define how the method should be identified and implemented in the base-class of stats

    def __call__(self, data, groupby, orient, scales):
        """
        Apply polynomial fitting and interpolation to grouped data.

        Parameters:
        - data: DataFrame with 'x' and 'y' columns.
        - groupby: Grouping object.
        - orient: Orientation of the plot.
        - scales: Scaling method.

        Returns:
        - DataFrame with interpolated 'x' and 'y' values for each group.
        """
        return (
            groupby
            .apply(data.dropna(subset=["x", "y"]), self._fit_predict)
        )
```