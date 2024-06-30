# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_species_kde.py`

```
"""
================================================
Kernel Density Estimate of Species Distributions
================================================
This shows an example of a neighbors-based query (in particular a kernel
density estimate) on geospatial data, using a Ball Tree built upon the
Haversine distance metric -- i.e. distances over points in latitude/longitude.
The dataset is provided by Phillips et. al. (2006).
If available, the example uses
`basemap <https://matplotlib.org/basemap/>`_
to plot the coast lines and national boundaries of South America.

This example does not perform any learning over the data
(see :ref:`sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py` for
an example of classification based on the attributes in this dataset).  It
simply shows the kernel density estimate of observed data points in
geospatial coordinates.

The two species are:

 - `"Bradypus variegatus"
   <https://www.iucnredlist.org/species/3038/47437046>`_ ,
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
"""  # noqa: E501

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from sklearn.datasets import fetch_species_distributions  # 导入fetch_species_distributions函数，用于获取物种分布数据集
from sklearn.neighbors import KernelDensity  # 导入KernelDensity类，用于核密度估计

# if basemap is available, we'll use it.
# otherwise, we'll improvise later...
try:
    from mpl_toolkits.basemap import Basemap  # 尝试导入Basemap类（如果可用）

    basemap = True  # 设置basemap为True，表示Basemap类可用
except ImportError:
    basemap = False  # 如果导入错误，则设置basemap为False，表示Basemap类不可用


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
    xmin = batch.x_left_lower_corner + batch.grid_size  # 计算x轴最小值
    xmax = xmin + (batch.Nx * batch.grid_size)  # 计算x轴最大值
    ymin = batch.y_left_lower_corner + batch.grid_size  # 计算y轴最小值
    ymax = ymin + (batch.Ny * batch.grid_size)  # 计算y轴最大值

    # x coordinates of the grid cells
    xgrid = np.arange(xmin, xmax, batch.grid_size)  # 生成x轴网格坐标数组
    # y coordinates of the grid cells
    ygrid = np.arange(ymin, ymax, batch.grid_size)  # 生成y轴网格坐标数组

    return (xgrid, ygrid)  # 返回x轴和y轴的网格坐标数组


# Get matrices/arrays of species IDs and locations
data = fetch_species_distributions()  # 获取物种分布数据集
species_names = ["Bradypus Variegatus", "Microryzomys Minutus"]  # 物种名称列表

Xtrain = np.vstack([data["train"]["dd lat"], data["train"]["dd long"]]).T  # 训练集的经纬度数据
ytrain = np.array(
    # 遍历"data"字典中"train"键对应的"species"列表中的每个元素"d"
    # 并对元素"d"进行ASCII解码，并检查解码后的字符串是否以"micro"开头
    [d.decode("ascii").startswith("micro") for d in data["train"]["species"]],
    # 将上述列表生成器生成的布尔值列表作为数据，创建一个NumPy整数类型的数组
    dtype="int",
# Convert latitude and longitude coordinates in degrees to radians for further calculations
Xtrain *= np.pi / 180.0

# Set up the data grid for the contour plot using the constructed grids from 'data'
xgrid, ygrid = construct_grids(data)
X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])  # Create a mesh grid with specific sampling intervals
land_reference = data.coverages[6][::5, ::5]  # Reference data for land cover at reduced resolution
land_mask = (land_reference > -9999).ravel()  # Create a mask for land areas excluding ocean

# Flatten coordinates and apply the land mask
xy = np.vstack([Y.ravel(), X.ravel()]).T
xy = xy[land_mask]
xy *= np.pi / 180.0  # Convert the selected coordinates to radians

# Create a new figure for plotting
fig = plt.figure()
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)

# Iterate over two species for plotting their distributions
for i in range(2):
    plt.subplot(1, 2, i + 1)

    # Print message indicating computation of Kernel Density Estimate (KDE) in spherical coordinates
    print(" - computing KDE in spherical coordinates")
    # Initialize KDE with specified parameters
    kde = KernelDensity(
        bandwidth=0.04, metric="haversine", kernel="gaussian", algorithm="ball_tree"
    )
    kde.fit(Xtrain[ytrain == i])  # Fit KDE for data of the current species

    # Initialize Z array for density values, set ocean values to -9999
    Z = np.full(land_mask.shape[0], -9999, dtype="int")
    Z[land_mask] = np.exp(kde.score_samples(xy))  # Compute density estimate for land points
    Z = Z.reshape(X.shape)  # Reshape Z to match the grid dimensions

    # Plot filled contours of the density estimate
    levels = np.linspace(0, Z.max(), 25)
    plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)

    # Depending on 'basemap' condition, plot coastlines either using Basemap or from coverage
    if basemap:
        print(" - plot coastlines using basemap")
        # Create Basemap object for geographical projections
        m = Basemap(
            projection="cyl",
            llcrnrlat=Y.min(),
            urcrnrlat=Y.max(),
            llcrnrlon=X.min(),
            urcrnrlon=X.max(),
            resolution="c",
        )
        m.drawcoastlines()  # Draw coastlines
        m.drawcountries()   # Draw country borders
    else:
        print(" - plot coastlines from coverage")
        # Plot coastlines from provided land reference data
        plt.contour(
            X, Y, land_reference, levels=[-9998], colors="k", linestyles="solid"
        )
        plt.xticks([])  # Disable x-axis ticks
        plt.yticks([])  # Disable y-axis ticks

    plt.title(species_names[i])  # Set plot title for the current species

plt.show()  # Display the entire plot
```