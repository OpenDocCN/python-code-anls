# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_tritools.py`

```
"""
Tools for triangular grids.
"""

import numpy as np  # 导入 NumPy 库

from matplotlib import _api  # 导入 Matplotlib 的 _api 模块
from matplotlib.tri import Triangulation  # 从 Matplotlib 的 tri 模块导入 Triangulation 类


class TriAnalyzer:
    """
    Define basic tools for triangular mesh analysis and improvement.

    A TriAnalyzer encapsulates a `.Triangulation` object and provides basic
    tools for mesh analysis and mesh improvement.

    Attributes
    ----------
    scale_factors : property
        Factors to rescale the triangulation into a unit square.

    Parameters
    ----------
    triangulation : `~matplotlib.tri.Triangulation`
        The encapsulated triangulation to analyze.
    """

    def __init__(self, triangulation):
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation  # 初始化 TriAnalyzer 对象的 Triangulation 属性

    @property
    def scale_factors(self):
        """
        Factors to rescale the triangulation into a unit square.

        Returns
        -------
        (float, float)
            Scaling factors (kx, ky) so that the triangulation
            ``[triangulation.x * kx, triangulation.y * ky]``
            fits exactly inside a unit square.
        """
        compressed_triangles = self._triangulation.get_masked_triangles()  # 获取压缩后的三角形索引
        node_used = (np.bincount(np.ravel(compressed_triangles),
                                 minlength=self._triangulation.x.size) != 0)  # 计算使用的节点索引
        return (1 / np.ptp(self._triangulation.x[node_used]),  # 计算 x 轴的缩放因子
                1 / np.ptp(self._triangulation.y[node_used]))  # 计算 y 轴的缩放因子
    def circle_ratios(self, rescale=True):
        """
        Return a measure of the triangulation triangles flatness.

        The ratio of the incircle radius over the circumcircle radius is a
        widely used indicator of a triangle's flatness.
        It is always ``<= 0.5`` and ``== 0.5`` only for equilateral
        triangles. Circle ratios below 0.01 denote very flat triangles.

        To avoid unduly low values due to a difference of scale between the 2
        axis, the triangular mesh can first be rescaled to fit inside a unit
        square with `scale_factors` (Only if *rescale* is True, which is
        its default value).

        Parameters
        ----------
        rescale : bool, default: True
            If True, internally rescale (based on `scale_factors`), so that the
            (unmasked) triangles fit exactly inside a unit square mesh.

        Returns
        -------
        masked array
            Ratio of the incircle radius over the circumcircle radius, for
            each 'rescaled' triangle of the encapsulated triangulation.
            Values corresponding to masked triangles are masked out.

        """
        # Coords rescaling based on `rescale` parameter
        if rescale:
            (kx, ky) = self.scale_factors  # Extract scaling factors from the object
        else:
            (kx, ky) = (1.0, 1.0)  # Default scaling factors
        # Stack x and y coordinates scaled by kx and ky respectively
        pts = np.vstack([self._triangulation.x*kx,
                         self._triangulation.y*ky]).T
        # Extract triangle points from scaled coordinates
        tri_pts = pts[self._triangulation.triangles]
        # Computes the lengths of the sides of each triangle
        a = tri_pts[:, 1, :] - tri_pts[:, 0, :]
        b = tri_pts[:, 2, :] - tri_pts[:, 1, :]
        c = tri_pts[:, 0, :] - tri_pts[:, 2, :]
        a = np.hypot(a[:, 0], a[:, 1])  # Length of side a
        b = np.hypot(b[:, 0], b[:, 1])  # Length of side b
        c = np.hypot(c[:, 0], c[:, 1])  # Length of side c
        # Calculate semi-perimeter s of each triangle
        s = (a+b+c)*0.5
        # Calculate the product used in circumcircle radius calculation
        prod = s*(a+b-s)*(a+c-s)*(b+c-s)
        # Handle triangles with infinite circumcircle radius (flat triangles)
        bool_flat = (prod == 0.)
        if np.any(bool_flat):
            # Pathological case: set infinite circumcircle radius for flat triangles
            ntri = tri_pts.shape[0]
            circum_radius = np.empty(ntri, dtype=np.float64)
            circum_radius[bool_flat] = np.inf
            abc = a*b*c
            circum_radius[~bool_flat] = abc[~bool_flat] / (
                4.0*np.sqrt(prod[~bool_flat]))
        else:
            # Normal case: calculate circumcircle radius for each triangle
            circum_radius = (a*b*c) / (4.0*np.sqrt(prod))
        # Calculate incircle radius for each triangle
        in_radius = (a*b*c) / (4.0*circum_radius*s)
        # Calculate circle ratio: ratio of incircle radius to circumcircle radius
        circle_ratio = in_radius/circum_radius
        # Apply mask from triangulation if present
        mask = self._triangulation.mask
        if mask is None:
            return circle_ratio  # Return circle ratios without mask
        else:
            return np.ma.array(circle_ratio, mask=mask)  # Return masked array of circle ratios
    def _get_compressed_triangulation(self):
        """
        Compress (if masked) the encapsulated triangulation.

        Returns minimal-length triangles array (*compressed_triangles*) and
        coordinates arrays (*compressed_x*, *compressed_y*) that can still
        describe the unmasked triangles of the encapsulated triangulation.

        Returns
        -------
        compressed_triangles : array-like
            the returned compressed triangulation triangles
        compressed_x : array-like
            the returned compressed triangulation 1st coordinate
        compressed_y : array-like
            the returned compressed triangulation 2nd coordinate
        tri_renum : int array
            renumbering table to translate the triangle numbers from the
            encapsulated triangulation into the new (compressed) renumbering.
            -1 for masked triangles (deleted from *compressed_triangles*).
        node_renum : int array
            renumbering table to translate the point numbers from the
            encapsulated triangulation into the new (compressed) renumbering.
            -1 for unused points (i.e. those deleted from *compressed_x* and
            *compressed_y*).

        """
        # 获取三角剖分的有效三角形和重编号
        tri_mask = self._triangulation.mask
        compressed_triangles = self._triangulation.get_masked_triangles()
        ntri = self._triangulation.triangles.shape[0]
        if tri_mask is not None:
            tri_renum = self._total_to_compress_renum(~tri_mask)
        else:
            tri_renum = np.arange(ntri, dtype=np.int32)

        # 获取有效节点和节点的重编号
        valid_node = (np.bincount(np.ravel(compressed_triangles),
                                  minlength=self._triangulation.x.size) != 0)
        compressed_x = self._triangulation.x[valid_node]
        compressed_y = self._triangulation.y[valid_node]
        node_renum = self._total_to_compress_renum(valid_node)

        # 对有效三角形节点进行重编号
        compressed_triangles = node_renum[compressed_triangles]

        return (compressed_triangles, compressed_x, compressed_y, tri_renum,
                node_renum)

    @staticmethod
    def _total_to_compress_renum(valid):
        """
        Parameters
        ----------
        valid : 1D bool array
            Validity mask.

        Returns
        -------
        int array
            Array so that (`valid_array` being a compressed array
            based on a `masked_array` with mask ~*valid*):

            - For all i with valid[i] = True:
              valid_array[renum[i]] = masked_array[i]
            - For all i with valid[i] = False:
              renum[i] = -1 (invalid value)
        """
        # 创建一个重编号数组，根据有效性掩码
        renum = np.full(np.size(valid), -1, dtype=np.int32)
        n_valid = np.sum(valid)
        renum[valid] = np.arange(n_valid, dtype=np.int32)
        return renum
```