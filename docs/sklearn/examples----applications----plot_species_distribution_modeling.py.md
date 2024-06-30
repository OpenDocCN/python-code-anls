# `D:\src\scipysrc\scikit-learn\examples\applications\plot_species_distribution_modeling.py`

```
"""
=============================
Species distribution modeling
=============================

Modeling species' geographic distributions is an important
problem in conservation biology. In this example, we
model the geographic distribution of two South American
mammals given past observations and 14 environmental
variables. Since we have only positive examples (there are
no unsuccessful observations), we cast this problem as a
density estimation problem and use the :class:`~sklearn.svm.OneClassSVM`
as our modeling tool. The dataset is provided by Phillips et. al. (2006).
If available, the example uses
`basemap <https://matplotlib.org/basemap/>`_
to plot the coast lines and national boundaries of South America.

The two species are:

 - `"Bradypus variegatus"
   <http://www.iucnredlist.org/details/3038/0>`_ ,
   the Brown-throated Sloth.

 - `"Microryzomys minutus"
   <http://www.iucnredlist.org/details/13408/0>`_ ,
   also known as the Forest Small Rice Rat, a rodent that lives in Peru,
   Colombia, Ecuador, Peru, and Venezuela.

References
----------

 * `"Maximum entropy modeling of species geographic distributions"
   <http://rob.schapire.net/papers/ecolmod.pdf>`_
   S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
   190:231-259, 2006.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics, svm
from sklearn.datasets import fetch_species_distributions
from sklearn.utils import Bunch

# if basemap is available, we'll use it.
# otherwise, we'll improvise later...
try:
    from mpl_toolkits.basemap import Basemap

    basemap = True
except ImportError:
    basemap = False


def construct_grids(batch):
    """Construct the map grid from the batch object

    Parameters
    ----------
    batch : Batch object
        The object returned by :func:`fetch_species_distributions`

    Returns
    -------
    (xgrid, ygrid) : 1-D arrays
        The grid corresponding to the values in batch.coverages
    """
    # x,y coordinates for corner cells
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + (batch.Nx * batch.grid_size)
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + (batch.Ny * batch.grid_size)

    # x coordinates of the grid cells
    xgrid = np.arange(xmin, xmax, batch.grid_size)
    # y coordinates of the grid cells
    ygrid = np.arange(ymin, ymax, batch.grid_size)

    return (xgrid, ygrid)


def create_species_bunch(species_name, train, test, coverages, xgrid, ygrid):
    """Create a bunch with information about a particular organism

    This will use the test/train record arrays to extract the
    data specific to the given species name.
    """
    # Create a Bunch object to store species information
    bunch = Bunch(name=" ".join(species_name.split("_")[:2]))
    # Encode species_name to ASCII
    species_name = species_name.encode("ascii")
    # Store test and train data in a dictionary under 'points'
    points = dict(test=test, train=train)
    # 遍历points字典中的每个标签和对应的点集
    for label, pts in points.items():
        # 选择与指定物种相关的点
        pts = pts[pts["species"] == species_name]
        # 将筛选后的点集存储到bunch字典中，键名为"pts_label"
        bunch["pts_%s" % label] = pts

        # 根据经纬度坐标在xgrid和ygrid中查找对应的索引
        ix = np.searchsorted(xgrid, pts["dd long"])
        iy = np.searchsorted(ygrid, pts["dd lat"])
        # 将对应的覆盖值存储到bunch字典中，键名为"cov_label"
        bunch["cov_%s" % label] = coverages[:, -iy, ix].T

    # 返回处理后的数据集
    return bunch
def plot_species_distribution(
    species=("bradypus_variegatus_0", "microryzomys_minutus_0")
):
    """
    Plot the species distribution.
    """
    # 如果提供的物种超过两种，打印警告信息
    if len(species) > 2:
        print(
            "Note: when more than two species are provided,"
            " only the first two will be used"
        )

    # 记录开始运行时间
    t0 = time()

    # 载入压缩的数据
    data = fetch_species_distributions()

    # 设置数据网格
    xgrid, ygrid = construct_grids(data)

    # 在 x,y 坐标上创建网格
    X, Y = np.meshgrid(xgrid, ygrid[::-1])

    # 为每个物种创建一个数据集合
    BV_bunch = create_species_bunch(
        species[0], data.train, data.test, data.coverages, xgrid, ygrid
    )
    MM_bunch = create_species_bunch(
        species[1], data.train, data.test, data.coverages, xgrid, ygrid
    )

    # 背景点（网格坐标）用于评估
    np.random.seed(13)
    background_points = np.c_[
        np.random.randint(low=0, high=data.Ny, size=10000),
        np.random.randint(low=0, high=data.Nx, size=10000),
    ].T

    # 我们将利用 coverages[6] 在所有陆地点上有测量的事实。
    # 这将帮助我们区分陆地和水域。
    land_reference = data.coverages[6]

    # 为每个物种拟合、预测并绘图。
    print("\ntime elapsed: %.2fs" % (time() - t0))


plot_species_distribution()
plt.show()
```