# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\intersecting_planes.py`

```py
"""
===================
Intersecting planes
===================

This examples demonstrates drawing intersecting planes in 3D. It is a generalization
of :doc:`/gallery/mplot3d/imshow3d`.

Drawing intersecting planes in `.mplot3d` is complicated, because `.mplot3d` is not a
real 3D renderer, but only projects the Artists into 3D and draws them in the right
order. This does not work correctly if Artists overlap each other mutually. In this
example, we lift the problem of mutual overlap by segmenting the planes at their
intersections, making four parts out of each plane.

This examples only works correctly for planes that cut each other in haves. This
limitation is intentional to keep the code more readable. Cutting at arbitrary
positions would of course be possible but makes the code even more complex.
Thus, this example is more a demonstration of the concept how to work around
limitations of the 3D visualization, it's not a refined solution for drawing
arbitrary intersecting planes, which you can copy-and-paste as is.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_quadrants(ax, array, fixed_coord, cmap):
    """
    For a given 3d *array* plot a plane with *fixed_coord*, using four quadrants.
    
    Parameters:
    - ax: Axes object for plotting.
    - array: 3D numpy array containing data for the planes.
    - fixed_coord: Coordinate ('x', 'y', or 'z') indicating which plane is fixed.
    - cmap: Colormap for coloring the planes.
    """
    # Determine the dimensions of the array
    nx, ny, nz = array.shape
    
    # Determine the slice indices based on the fixed coordinate
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    
    # Extract data for the plane at the fixed coordinate
    plane_data = array[index]
    
    # Divide the plane into four quadrants
    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]
    
    # Determine the minimum and maximum values in the entire array
    min_val = array.min()
    max_val = array.max()
    
    # Get the colormap for coloring the planes
    cmap = plt.get_cmap(cmap)
    
    # Iterate over the quadrants and plot each one
    for i, quadrant in enumerate(quadrants):
        # Determine the face colors based on the quadrant's data
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        
        # Plot the surface of the quadrant based on the fixed coordinate
        if fixed_coord == 'x':
            Y, Z = np.mgrid[0:ny // 2, 0:nz // 2]
            X = nx // 2 * np.ones_like(Y)
            Y_offset = (i // 2) * ny // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'y':
            X, Z = np.mgrid[0:nx // 2, 0:nz // 2]
            Y = ny // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'z':
            X, Y = np.mgrid[0:nx // 2, 0:ny // 2]
            Z = nz // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Y_offset = (i % 2) * ny // 2
            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)


def figure_3D_array_slices(array, cmap=None):
    """
    Plot intersecting planes of a 3D array using matplotlib.
    
    Parameters:
    - array: 3D numpy array containing data for the planes.
    - cmap: Colormap for coloring the planes.
    """
    """Plot a 3d array using three intersecting centered planes."""
    # 创建一个新的 3D 图形对象
    fig = plt.figure()
    # 在图形上添加一个 3D 子图
    ax = fig.add_subplot(projection='3d')
    # 设置图形的盒子长宽高比例，以数组的形状为基础
    ax.set_box_aspect(array.shape)
    # 在三个不同的平面上绘制数据数组的四分之一部分，分别以 x、y、z 方向为中心平面
    plot_quadrants(ax, array, 'x', cmap=cmap)
    plot_quadrants(ax, array, 'y', cmap=cmap)
    plot_quadrants(ax, array, 'z', cmap=cmap)
    # 返回绘制好的图形对象和子图对象
    return fig, ax
# 定义三维数组的维度
nx, ny, nz = 70, 100, 50
# 使用 numpy 的 mgrid 函数生成一个三维坐标系的平方和数组
# np.mgrid[-1:1:1j*nx, -1:1:1j*ny, -1:1:1j*nz] 生成三个长度为 nx, ny, nz 的数组，代表三维空间内坐标值
# ** 2 对这些坐标值每个分量进行平方操作
# .sum(0) 对三维坐标每个位置进行求和，得到一个二维数组
r_square = (np.mgrid[-1:1:1j * nx, -1:1:1j * ny, -1:1:1j * nz] ** 2).sum(0)

# 调用 figure_3D_array_slices 函数，绘制三维数组的切片图像
# r_square 是要绘制的数组数据，cmap='viridis_r' 指定色彩映射为 viridis_r（反色的 viridis 色彩映射）
plt.show()
```