# `D:\src\scipysrc\sympy\sympy\categories\diagram_drawing.py`

```
r"""
This module contains the functionality to arrange the nodes of a
diagram on an abstract grid, and then to produce a graphical
representation of the grid.

The currently supported back-ends are Xy-pic [Xypic].

Layout Algorithm
================

This section provides an overview of the algorithms implemented in
:class:`DiagramGrid` to lay out diagrams.

The first step of the algorithm is the removal composite and identity
morphisms which do not have properties in the supplied diagram.  The
premises and conclusions of the diagram are then merged.

The generic layout algorithm begins with the construction of the
"skeleton" of the diagram.  The skeleton is an undirected graph which
has the objects of the diagram as vertices and has an (undirected)
edge between each pair of objects between which there exist morphisms.
The direction of the morphisms does not matter at this stage.  The
skeleton also includes an edge between each pair of vertices `A` and
`C` such that there exists an object `B` which is connected via
a morphism to `A`, and via a morphism to `C`.

The skeleton constructed in this way has the property that every
object is a vertex of a triangle formed by three edges of the
skeleton.  This property lies at the base of the generic layout
algorithm.

After the skeleton has been constructed, the algorithm lists all
triangles which can be formed.  Note that some triangles will not have
all edges corresponding to morphisms which will actually be drawn.
Triangles which have only one edge or less which will actually be
drawn are immediately discarded.

The list of triangles is sorted according to the number of edges which
correspond to morphisms, then the triangle with the least number of such
edges is selected.  One of such edges is picked and the corresponding
objects are placed horizontally, on a grid.  This edge is recorded to
be in the fringe.  The algorithm then finds a "welding" of a triangle
to the fringe.  A welding is an edge in the fringe where a triangle
could be attached.  If the algorithm succeeds in finding such a
welding, it adds to the grid that vertex of the triangle which was not
yet included in any edge in the fringe and records the two new edges in
the fringe.  This process continues iteratively until all objects of
the diagram has been placed or until no more weldings can be found.

An edge is only removed from the fringe when a welding to this edge
has been found, and there is no room around this edge to place
another vertex.

When no more weldings can be found, but there are still triangles
left, the algorithm searches for a possibility of attaching one of the
remaining triangles to the existing structure by a vertex.  If such a
possibility is found, the corresponding edge of the found triangle is
placed in the found space and the iterative process of welding
triangles restarts.

When logical groups are supplied, each of these groups is laid out
independently.  Then a diagram is constructed in which groups are
"""
"""
objects and any two logical groups between which there exist morphisms
are connected via a morphism.  This diagram is laid out.  Finally,
the grid which includes all objects of the initial diagram is
constructed by replacing the cells which contain logical groups with
the corresponding laid out grids, and by correspondingly expanding the
rows and columns.

The sequential layout algorithm begins by constructing the
underlying undirected graph defined by the morphisms obtained after
simplifying premises and conclusions and merging them (see above).
The vertex with the minimal degree is then picked up and depth-first
search is started from it.  All objects which are located at distance
`n` from the root in the depth-first search tree, are positioned in
the `n`-th column of the resulting grid.  The sequential layout will
therefore attempt to lay the objects out along a line.

References
==========

.. [Xypic] https://xy-pic.sourceforge.net/

"""

from sympy.categories import (CompositeMorphism, IdentityMorphism,
                              NamedMorphism, Diagram)
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on

from itertools import chain


__doctest_requires__ = {('preview_diagram',): 'pyglet'}


class _GrowableGrid:
    """
    Holds a growable grid of objects.

    Explanation
    ===========

    It is possible to append or prepend a row or a column to the grid
    using the corresponding methods.  Prepending rows or columns has
    the effect of changing the coordinates of the already existing
    elements.

    This class currently represents a naive implementation of the
    functionality with little attempt at optimisation.
    """
    
    def __init__(self, width, height):
        """
        Initialize a growable grid with the specified width and height.

        Parameters
        ----------
        width : int
            The initial width of the grid.
        height : int
            The initial height of the grid.
        """
        self._width = width
        self._height = height

        # Initialize the 2D array with None values
        self._array = [[None for j in range(width)] for i in range(height)]

    @property
    def width(self):
        """
        Get the current width of the grid.

        Returns
        -------
        int
            The width of the grid.
        """
        return self._width

    @property
    def height(self):
        """
        Get the current height of the grid.

        Returns
        -------
        int
            The height of the grid.
        """
        return self._height

    def __getitem__(self, i_j):
        """
        Get the element located at the specified (i, j) position in the grid.

        Parameters
        ----------
        i_j : tuple of int
            Coordinates (i, j) of the element.

        Returns
        -------
        object or None
            The element at position (i, j).
        """
        i, j = i_j
        return self._array[i][j]

    def __setitem__(self, i_j, newvalue):
        """
        Set the element located at the specified (i, j) position in the grid.

        Parameters
        ----------
        i_j : tuple of int
            Coordinates (i, j) of the element to set.
        newvalue : object
            New value to set at position (i, j).
        """
        i, j = i_j
        self._array[i][j] = newvalue

    def append_row(self):
        """
        Append an empty row to the grid.
        """
        self._height += 1
        self._array.append([None for j in range(self._width)])

    def append_column(self):
        """
        Append an empty column to the grid.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].append(None)
    # 在网格的顶部添加一行空行
    def prepend_row(self):
        """
        Prepends the grid with an empty row.
        """
        # 增加网格的高度
        self._height += 1
        # 在数组的最前面插入一个宽度为self._width的空行
        self._array.insert(0, [None for j in range(self._width)])

    # 在网格的左侧添加一列空列
    def prepend_column(self):
        """
        Prepends the grid with an empty column.
        """
        # 增加网格的宽度
        self._width += 1
        # 遍历每一行，将空值(None)插入到每行的最前面
        for i in range(self._height):
            self._array[i].insert(0, None)
class DiagramGrid:
    r"""
    Constructs and holds the fitting of the diagram into a grid.

    Explanation
    ===========

    The mission of this class is to analyse the structure of the
    supplied diagram and to place its objects on a grid such that,
    when the objects and the morphisms are actually drawn, the diagram
    would be "readable", in the sense that there will not be many
    intersections of moprhisms.  This class does not perform any
    actual drawing.  It does strive nevertheless to offer sufficient
    metadata to draw a diagram.

    Consider the following simple diagram.

    >>> from sympy.categories import Object, NamedMorphism
    >>> from sympy.categories import Diagram, DiagramGrid
    >>> from sympy import pprint
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g])

    The simplest way to have a diagram laid out is the following:

    >>> grid = DiagramGrid(diagram)
    >>> (grid.width, grid.height)
    (2, 2)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C

    Sometimes one sees the diagram as consisting of logical groups.
    One can advise ``DiagramGrid`` as to such groups by employing the
    ``groups`` keyword argument.

    Consider the following diagram:

    >>> D = Object("D")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])

    Lay it out with generic layout:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B  D
    <BLANKLINE>
       C

    Now, we can group the objects `A` and `D` to have them near one
    another:

    >>> grid = DiagramGrid(diagram, groups=[[A, D], B, C])
    >>> pprint(grid)
    B     C
    <BLANKLINE>
    A  D

    Note how the positioning of the other objects changes.

    Further indications can be supplied to the constructor of
    :class:`DiagramGrid` using keyword arguments.  The currently
    supported hints are explained in the following paragraphs.

    :class:`DiagramGrid` does not automatically guess which layout
    would suit the supplied diagram better.  Consider, for example,
    the following linear diagram:

    >>> E = Object("E")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(C, D, "h")
    >>> i = NamedMorphism(D, E, "i")
    >>> diagram = Diagram([f, g, h, i])

    When laid out with the generic layout, it does not get to look
    linear:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C  D
    <BLANKLINE>
          E

    To get it laid out in a line, use ``layout="sequential"``:

    >>> grid = DiagramGrid(diagram, layout="sequential")
    >>> pprint(grid)
    A  B  C  D  E

    One may sometimes need to transpose the resulting layout.  While
    """
    
    def __init__(self, diagram, groups=None, layout="generic"):
        r"""
        Initialize the DiagramGrid object with a given diagram and optional layout hints.

        Parameters:
        -----------
        diagram : Diagram
            The diagram object containing morphisms and objects.
        groups : list of lists, optional
            Lists of objects that should be grouped together in the layout.
        layout : str, optional
            Specifies the layout strategy ('generic' or 'sequential').

        Returns:
        --------
        None

        Notes:
        ------
        This constructor initializes the grid layout for the diagram based on the given parameters.
        """
        pass
    """
    this can always be done by hand, :class:`DiagramGrid` provides a
    hint for that purpose:
    """

    @staticmethod
    def _simplify_morphisms(morphisms):
        """
        Given a dictionary mapping morphisms to their properties,
        returns a new dictionary in which there are no morphisms which
        do not have properties, and which are compositions of other
        morphisms included in the dictionary.  Identities are dropped
        as well.
        """
        newmorphisms = {}
        for morphism, props in morphisms.items():
            if isinstance(morphism, CompositeMorphism) and not props:
                # Skip morphisms that are compositions without properties
                continue
            elif isinstance(morphism, IdentityMorphism):
                # Skip identity morphisms
                continue
            else:
                # Add valid morphisms to the new dictionary
                newmorphisms[morphism] = props
        return newmorphisms

    @staticmethod
    def _merge_premises_conclusions(premises, conclusions):
        """
        Given two dictionaries of morphisms and their properties,
        produces a single dictionary which includes elements from both
        dictionaries.  If a morphism has some properties in premises
        and also in conclusions, the properties in conclusions take
        priority.
        """
        return dict(chain(premises.items(), conclusions.items()))

    @staticmethod
    def _juxtapose_edges(edge1, edge2):
        """
        If ``edge1`` and ``edge2`` have precisely one common endpoint,
        returns an edge which would form a triangle with ``edge1`` and
        ``edge2``.

        If ``edge1`` and ``edge2`` do not have a common endpoint,
        returns ``None``.

        If ``edge1`` and ``edge`` are the same edge, returns ``None``.
        """
        intersection = edge1 & edge2
        if len(intersection) != 1:
            # Return None if edges have no common points or are identical
            return None

        # Construct a new edge from the unique endpoints of edge1 and edge2
        return (edge1 - intersection) | (edge2 - intersection)
    def _add_edge_append(dictionary, edge, elem):
        """
        如果 ``edge`` 不在 ``dictionary`` 中，则将 ``edge`` 添加到字典中，并将其值设为 ``[elem]``。否则将 ``elem`` 追加到现有条目的值中。

        注意，边是无向的，因此 ``(A, B) = (B, A)``。
        """
        if edge in dictionary:
            # 如果边已经存在于字典中，则将元素追加到对应的值列表中
            dictionary[edge].append(elem)
        else:
            # 如果边不存在于字典中，则创建一个新的条目
            dictionary[edge] = [elem]

    @staticmethod
    def _build_skeleton(morphisms):
        """
        创建一个将边映射到相应态射的字典。因此对于态射 `f:A\rightarrow B`，边 `(A, B)` 将与 `f` 关联。此函数还将通过并置两个已在列表中的边形成的新边添加到列表中。这些新边不与任何态射相关联，只是为了确保图表可以分解为三角形。
        """
        edges = {}
        # 为态射创建边。
        for morphism in morphisms:
            DiagramGrid._add_edge_append(
                edges, frozenset([morphism.domain, morphism.codomain]), morphism)

        # 通过并置现有边创建新边。
        edges1 = dict(edges)
        for w in edges1:
            for v in edges1:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv not in edges:
                    edges[wv] = []

        return edges

    @staticmethod
    def _list_triangles(edges):
        """
        构建由提供的边形成的三角形集合。这些三角形是任意的，不需要是可交换的。三角形是一个集合，其中包含其三条边。
        """
        triangles = set()

        for w in edges:
            for v in edges:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv in edges:
                    triangles.add(frozenset([w, v, wv]))

        return triangles

    @staticmethod
    def _drop_redundant_triangles(triangles, skeleton):
        """
        返回一个列表，其中仅包含那些至少与两条边相关联的三角形。
        """
        return [tri for tri in triangles
                if len([e for e in tri if skeleton[e]]) >= 2]

    @staticmethod
    def _morphism_length(morphism):
        """
        返回态射的长度。态射的长度是它包含的组件数量。非复合态射的长度为1。
        """
        if isinstance(morphism, CompositeMorphism):
            return len(morphism.components)
        else:
            return 1
    def _compute_triangle_min_sizes(triangles, edges):
        r"""
        Returns a dictionary mapping triangles to their minimal sizes.
        The minimal size of a triangle is the sum of maximal lengths
        of morphisms associated to the sides of the triangle.  The
        length of a morphism is the number of components it consists
        of.  A non-composite morphism is of length 1.

        Sorting triangles by this metric attempts to address two
        aspects of layout.  For triangles with only simple morphisms
        in the edge, this assures that triangles with all three edges
        visible will get typeset after triangles with less visible
        edges, which sometimes minimizes the necessity in diagonal
        arrows.  For triangles with composite morphisms in the edges,
        this assures that objects connected with shorter morphisms
        will be laid out first, resulting the visual proximity of
        those objects which are connected by shorter morphisms.
        """
        # 初始化一个空字典来存储每个三角形的最小尺寸
        triangle_sizes = {}
        # 遍历每个三角形
        for triangle in triangles:
            size = 0
            # 遍历三角形的每条边
            for e in triangle:
                # 获取边 e 对应的所有态射
                morphisms = edges[e]
                # 如果存在态射，则计算其最大长度的和作为该三角形的尺寸
                if morphisms:
                    size += max(DiagramGrid._morphism_length(m)
                                for m in morphisms)
            # 将三角形和其计算出的尺寸存入字典中
            triangle_sizes[triangle] = size
        return triangle_sizes

    @staticmethod
    def _triangle_objects(triangle):
        """
        Given a triangle, returns the objects included in it.
        """
        # 三角形被表示为一个 frozenset，包含三条边（每条边是一个包含两个元素的 frozenset）
        # 使用 chain(*tuple(triangle)) 将三个边连接起来，并创建一个新的 frozenset，表示三角形中的所有对象
        return frozenset(chain(*tuple(triangle)))

    @staticmethod
    def _other_vertex(triangle, edge):
        """
        Given a triangle and an edge of it, returns the vertex which
        opposes the edge.
        """
        # 获取三角形中的所有对象集合，并从中减去边 `edge` 中的对象集合，以获得对立的顶点
        return list(DiagramGrid._triangle_objects(triangle) - set(edge))[0]

    @staticmethod
    def _empty_point(pt, grid):
        """
        Checks if the cell at coordinates ``pt`` is either empty or
        out of the bounds of the grid.
        """
        # 检查坐标为 `pt` 的单元格是否为空或超出网格边界
        if (pt[0] < 0) or (pt[1] < 0) or \
           (pt[0] >= grid.height) or (pt[1] >= grid.width):
            return True
        # 返回该坐标在网格中对应的值是否为空
        return grid[pt] is None
    def _put_object(coords, obj, grid, fringe):
        """
        Places an object at the coordinate ``coords`` in ``grid``,
        growing the grid and updating ``fringe``, if necessary.
        Returns (0, 0) if no row or column has been prepended, (1, 0)
        if a row was prepended, (0, 1) if a column was prepended, and
        (1, 1) if both a column and a row were prepended.
        """
        (i, j) = coords  # 解构坐标元组，获取行和列索引
        offset = (0, 0)  # 初始化偏移量为(0, 0)

        if i == -1:
            # 如果行索引为-1，表示需要在顶部增加一行
            grid.prepend_row()
            i = 0  # 更新行索引为0
            offset = (1, 0)  # 设置偏移量为(1, 0)，表示增加了一行
            # 更新 fringe 中的每个坐标对，使其行索引均加1
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1 + 1, j1), (i2 + 1, j2))
        elif i == grid.height:
            # 如果行索引等于 grid 的高度，需要在底部增加一行
            grid.append_row()

        if j == -1:
            # 如果列索引为-1，表示需要在左侧增加一列
            j = 0  # 更新列索引为0
            offset = (offset[0], 1)  # 更新偏移量为(0, 1)，表示增加了一列
            grid.prepend_column()
            # 更新 fringe 中的每个坐标对，使其列索引均加1
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1, j1 + 1), (i2, j2 + 1))
        elif j == grid.width:
            # 如果列索引等于 grid 的宽度，需要在右侧增加一列
            grid.append_column()

        # 在 grid 的指定坐标处放置 obj
        grid[i, j] = obj
        return offset



    @staticmethod
    def _choose_target_cell(pt1, pt2, edge, obj, skeleton, grid):
        """
        Given two points, ``pt1`` and ``pt2``, and the welding edge
        ``edge``, chooses one of the two points to place the opposing
        vertex ``obj`` of the triangle. If neither of these points
        fits, returns ``None``.
        """
        # 检查 pt1 和 pt2 是否为空单元格
        pt1_empty = DiagramGrid._empty_point(pt1, grid)
        pt2_empty = DiagramGrid._empty_point(pt2, grid)

        if pt1_empty and pt2_empty:
            # 如果两个单元格都为空，则选择其中一个单元格，确保三角形的一个可见边与当前焊接边垂直
            A = grid[edge[0]]  # 获取与边 edge[0] 相关联的对象 A

            if skeleton.get(frozenset([A, obj])):
                # 如果三角形的一个可见边与当前焊接边垂直，则选择 pt1
                return pt1
            else:
                # 否则选择 pt2
                return pt2
        if pt1_empty:
            # 如果只有 pt1 为空，则选择 pt1
            return pt1
        elif pt2_empty:
            # 如果只有 pt2 为空，则选择 pt2
            return pt2
        else:
            # 如果两个单元格都不为空，则返回 None
            return None



    @staticmethod
    def _find_triangle_to_weld(triangles, fringe, grid):
        """
        Finds, if possible, a triangle and an edge in the ``fringe`` to
        which the triangle could be attached. Returns the tuple
        containing the triangle and the index of the corresponding
        edge in the ``fringe``.

        This function relies on the fact that objects are unique in
        the diagram.
        """
        # 遍历所有三角形
        for triangle in triangles:
            # 遍历 fringe 中的每个边
            for (a, b) in fringe:
                # 如果 grid 中与边 (a, b) 相关联的对象集合在 triangle 中
                if frozenset([grid[a], grid[b]]) in triangle:
                    # 返回包含三角形和 fringe 中对应边索引的元组
                    return (triangle, (a, b))
        # 如果找不到可连接的三角形，则返回 None
        return None
    @staticmethod
    def _triangle_key(tri, triangle_sizes):
        """
        Returns a key for the supplied triangle.  It should be the
        same independently of the hash randomisation.
        """
        # 获取三角形中的对象并按照默认排序键排序
        objects = sorted(
            DiagramGrid._triangle_objects(tri), key=default_sort_key)
        # 返回一个元组作为三角形的键，包括三角形大小和对象的排序键
        return (triangle_sizes[tri], default_sort_key(objects))

    @staticmethod
    def _pick_root_edge(tri, skeleton):
        """
        For a given triangle always picks the same root edge.  The
        root edge is the edge that will be placed first on the grid.
        """
        # 从三角形中选取根边，根边是将首先放置在网格上的边
        candidates = [sorted(e, key=default_sort_key)
                      for e in tri if skeleton[e]]
        # 对候选边进行排序
        sorted_candidates = sorted(candidates, key=default_sort_key)
        # 返回排序后的第一个候选边，确保边上顶点的正确顺序
        return tuple(sorted(sorted_candidates[0], key=default_sort_key))

    @staticmethod
    def _drop_irrelevant_triangles(triangles, placed_objects):
        """
        Returns only those triangles whose set of objects is not
        completely included in ``placed_objects``.
        """
        # 返回那些对象集合不完全包含在 placed_objects 中的三角形列表
        return [tri for tri in triangles if not placed_objects.issuperset(
            DiagramGrid._triangle_objects(tri))]

    @staticmethod
    def _get_undirected_graph(objects, merged_morphisms):
        """
        Given the objects and the relevant morphisms of a diagram,
        returns the adjacency lists of the underlying undirected
        graph.
        """
        # 创建对象到邻接对象列表的字典
        adjlists = {obj: [] for obj in objects}

        # 根据合并的态射构建无向图的邻接列表
        for morphism in merged_morphisms:
            adjlists[morphism.domain].append(morphism.codomain)
            adjlists[morphism.codomain].append(morphism.domain)

        # 确保邻接列表中的对象始终以相同的顺序排列
        for obj in adjlists.keys():
            adjlists[obj].sort(key=default_sort_key)

        # 返回无向图的邻接列表
        return adjlists
    @staticmethod
    def _drop_inessential_morphisms(merged_morphisms):
        r"""
        Removes those morphisms which should appear in the diagram,
        but which have no relevance to object layout.
        
        Currently this removes "loop" morphisms: the non-identity
        morphisms with the same domains and codomains.
        """
        # 筛选出在图表中没有影响的形态态射，目前包括"循环"形态态射：具有相同定义域和值域的非恒等态射。
        morphisms = [m for m in merged_morphisms if m.domain != m.codomain]
        # 返回筛选后的形态态射列表
        return morphisms
    # 定义一个函数，用于获取给定对象集合和合并态射的连接组件列表
    def _get_connected_components(objects, merged_morphisms):
        """
        Given a container of morphisms, returns a list of connected
        components formed by these morphisms.  A connected component
        is represented by a diagram consisting of the corresponding
        morphisms.
        """
        # 初始化一个空字典，用于记录每个对象所属的组件索引，初始为None
        component_index = {}
        for o in objects:
            component_index[o] = None

        # 调用DiagramGrid类的方法，获取图表的无向图表示
        adjlist = DiagramGrid._get_undirected_graph(objects, merged_morphisms)

        # 定义一个深度优先搜索函数，用于遍历包含特定对象的组件
        def traverse_component(object, current_index):
            """
            Does a depth-first search traversal of the component
            containing ``object``.
            """
            # 将当前对象标记为属于当前索引的组件
            component_index[object] = current_index
            # 遍历当前对象的邻接对象
            for o in adjlist[object]:
                # 如果邻接对象尚未被标记为属于任何组件，则递归调用遍历函数
                if component_index[o] is None:
                    traverse_component(o, current_index)

        # 遍历所有的组件
        current_index = 0
        for o in adjlist:
            # 如果对象尚未被标记为属于任何组件，则从该对象开始遍历其组件
            if component_index[o] is None:
                traverse_component(o, current_index)
                current_index += 1

        # 按组件索引整理对象
        component_objects = [[] for i in range(current_index)]
        for o, idx in component_index.items():
            component_objects[idx].append(o)

        # 最后，整理每个组件中的态射
        #
        # 注意：如果有孤立的对象，在此阶段它们将不会有任何态射。
        # 由于布局算法依赖这些态射，我们需要为这些孤立的对象提供平凡的单位态射。
        # 这些态射在后续会被丢弃，但对象将会保留。

        component_morphisms = []
        for component in component_objects:
            current_morphisms = {}
            for m in merged_morphisms:
                # 如果态射的定义域和值域都在当前组件中，则将其添加到当前组件的态射集合中
                if (m.domain in component) and (m.codomain in component):
                    current_morphisms[m] = merged_morphisms[m]

            # 如果组件只包含一个对象，则添加一个单位态射以确保该组件中有态射存在
            if len(component) == 1:
                current_morphisms[IdentityMorphism(component[0])] = FiniteSet()

            # 将当前组件的态射集合转换为Diagram对象并添加到组件列表中
            component_morphisms.append(Diagram(current_morphisms))

        # 返回所有组件的列表
        return component_morphisms
    # 构造函数，初始化对象实例时调用，接受一个图表对象和一些可选的组和提示信息
    def __init__(self, diagram, groups=None, **hints):
        # 简化图表的前提条件中的态射
        premises = DiagramGrid._simplify_morphisms(diagram.premises)
        # 简化图表的结论中的态射
        conclusions = DiagramGrid._simplify_morphisms(diagram.conclusions)
        # 合并简化后的前提和结论中的态射
        all_merged_morphisms = DiagramGrid._merge_premises_conclusions(
            premises, conclusions)
        # 从合并后的态射中去除不必要的态射
        merged_morphisms = DiagramGrid._drop_inessential_morphisms(
            all_merged_morphisms)

        # 将所有合并的态射存储起来，以备后续使用
        self._morphisms = all_merged_morphisms

        # 获取连接的组件
        components = DiagramGrid._get_connected_components(
            diagram.objects, all_merged_morphisms)

        if groups and (groups != diagram.objects):
            # 如果有指定组，并且不等于图表中的对象集合，则根据组进行布局
            self._grid = DiagramGrid._handle_groups(
                diagram, groups, merged_morphisms, hints)
        elif len(components) > 1:
            # 如果图表有多个连接的组件

            # 注意：我们在检查布局提示之前就检查连接性，因为布局策略不知道如何处理不连通的图表。

            # 图表是不连通的。独立地布局各个组件。
            grids = []

            # 对组件进行排序，以确保最终以固定的、与哈希无关的顺序排列网格
            components = sorted(components, key=default_sort_key)

            for component in components:
                # 为每个组件创建一个图表网格
                grid = DiagramGrid(component, **hints)
                grids.append(grid)

            # 将各个网格按行连接在一起
            total_width = sum(g.width for g in grids)
            total_height = max(g.height for g in grids)

            grid = _GrowableGrid(total_width, total_height)
            start_j = 0
            for g in grids:
                for i in range(g.height):
                    for j in range(g.width):
                        grid[i, start_j + j] = g[i, j]

                start_j += g.width

            self._grid = grid
        elif "layout" in hints:
            if hints["layout"] == "sequential":
                # 如果有指定布局为"sequential"，则采用顺序布局
                self._grid = DiagramGrid._sequential_layout(
                    diagram, merged_morphisms)
        else:
            # 否则采用通用布局
            self._grid = DiagramGrid._generic_layout(diagram, merged_morphisms)

        if hints.get("transpose"):
            # 如果提示中有"transpose"标志，则对结果网格进行转置操作
            grid = _GrowableGrid(self._grid.height, self._grid.width)
            for i in range(self._grid.height):
                for j in range(self._grid.width):
                    grid[j, i] = self._grid[i, j]
            self._grid = grid
    def width(self):
        """
        Returns the number of columns in this diagram layout.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.width
        2

        """
        # 返回当前图表布局中列的数量
        return self._grid.width

    @property
    def height(self):
        """
        Returns the number of rows in this diagram layout.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.height
        2

        """
        # 返回当前图表布局中行的数量
        return self._grid.height

    def __getitem__(self, i_j):
        """
        Returns the object placed in the row ``i`` and column ``j``.
        The indices are 0-based.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> (grid[0, 0], grid[0, 1])
        (Object("A"), Object("B"))
        >>> (grid[1, 0], grid[1, 1])
        (None, Object("C"))

        """
        # 返回位于行 ``i`` 和列 ``j`` 的对象。索引从0开始。
        i, j = i_j
        return self._grid[i, j]

    @property
    def morphisms(self):
        """
        Returns those morphisms (and their properties) which are
        sufficiently meaningful to be drawn.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> grid.morphisms
        {NamedMorphism(Object("A"), Object("B"), "f"): EmptySet,
        NamedMorphism(Object("B"), Object("C"), "g"): EmptySet}

        """
        # 返回那些足够有意义以进行绘制的态射及其属性。
        return self._morphisms
    # 返回当前对象的字符串表示形式
    def __str__(self):
        """
        Produces a string representation of this class.

        This method returns a string representation of the underlying
        list of lists of objects.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import Diagram, DiagramGrid
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> diagram = Diagram([f, g])
        >>> grid = DiagramGrid(diagram)
        >>> print(grid)
        [[Object("A"), Object("B")],
        [None, Object("C")]]

        """
        # 返回对象内部网格数组的字符串表示形式
        return repr(self._grid._array)
# 定义描述箭头字符串的类
class ArrowStringDescription:
    r"""
    存储生成 Xy-pic 描述箭头所需的信息。

    该类的主要目标是抽象出箭头的字符串表示，并提供生成实际 Xy-pic 字符串的功能。

    ``unit`` 设置用于指定曲线和其他距离的单位。``horizontal_direction`` 应为 ``"r"`` 或 ``"l"`` 字符串，
    指定箭头目标单元格相对于当前单元格的水平偏移量。
    ``vertical_direction`` 应使用一系列 ``"d"`` 或 ``"u"`` 指定垂直偏移量。
    ``label_position`` 应为 ``"^"``, ``"_"``, 或 ``"|"`` 中的一个，指定标签应放置在箭头的上方、下方或箭头上方的断裂处。
    注意，“上方”和“下方”是相对于箭头方向而言的。``label`` 存储态射标签。

    以下是示例（忽略尚未解释的参数）：

    >>> from sympy.categories.diagram_drawing import ArrowStringDescription
    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \ar[dr]_{f}

    ``curving`` 应为 ``"^"``, ``"_"`` 中的一个，指定箭头弯曲的方向。
    ``curving_amount`` 是描述态射弯曲多少个 ``unit`` 的数字：

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> print(str(astr))
    \ar@/^12mm/[dr]_{f}

    ``looping_start`` 和 ``looping_end`` 目前仅用于循环态射，即起点和终点相同的态射。
    这两个属性应存储有效的 Xy-pic 方向，并分别指定箭头向外延伸和向内返回的方向：

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving=None, curving_amount=None,
    ... looping_start="u", looping_end="l", horizontal_direction="",
    ... vertical_direction="", label_position="_", label="f")
    >>> print(str(astr))
    \ar@(u,l)[]_{f}

    ``label_displacement`` 控制箭头标签距离箭头端点的距离。例如，将箭头标签位置靠近箭头头部，使用 ">":

    >>> astr = ArrowStringDescription(
    ... unit="mm", curving="^", curving_amount=12,
    ... looping_start=None, looping_end=None, horizontal_direction="d",
    ... vertical_direction="r", label_position="_", label="f")
    >>> astr.label_displacement = ">"
    >>> print(str(astr))
    \ar@/^12mm/[dr]_>{f}
    # 定义一个箭头描述类，用于生成 LaTeX Xy-pic 图形中的箭头描述字符串
    class ArrowStringDescription:
        def __init__(self, unit, curving, curving_amount, looping_start,
                     looping_end, horizontal_direction, vertical_direction,
                     label_position, label):
            # 设置箭头单位
            self.unit = unit
            # 设置曲线方向
            self.curving = curving
            # 设置曲线弯曲量
            self.curving_amount = curving_amount
            # 设置循环起点
            self.looping_start = looping_start
            # 设置循环终点
            self.looping_end = looping_end
            # 设置水平方向
            self.horizontal_direction = horizontal_direction
            # 设置垂直方向
            self.vertical_direction = vertical_direction
            # 设置标签位置
            self.label_position = label_position
            # 设置标签内容
            self.label = label
    
            # 初始化标签位移为空字符串
            self.label_displacement = ""
            # 初始化箭头样式为空字符串
            self.arrow_style = ""
    
            # 标记标签位置是否在设置曲线箭头时被固定，不应后续修改
            self.forced_label_position = False
    
        # 返回箭头描述字符串的方法
        def __str__(self):
            # 构造曲线描述字符串
            if self.curving:
                curving_str = "@/%s%d%s/" % (self.curving, self.curving_amount,
                                             self.unit)
            else:
                curving_str = ""
    
            # 构造循环描述字符串
            if self.looping_start and self.looping_end:
                looping_str = "@(%s,%s)" % (self.looping_start, self.looping_end)
            else:
                looping_str = ""
    
            # 构造箭头样式描述字符串
            if self.arrow_style:
                style_str = "@" + self.arrow_style
            else:
                style_str = ""
    
            # 构造完整的箭头描述字符串并返回
            return "\\ar%s%s%s[%s%s]%s%s{%s}" % \
                   (curving_str, looping_str, style_str, self.horizontal_direction,
                    self.vertical_direction, self.label_position,
                    self.label_displacement, self.label)
class XypicDiagramDrawer:
    r"""
    给定一个 :class:`~.Diagram` 和相应的 :class:`DiagramGrid`，生成图表的 Xy-pic 表示。

    本类中最重要的方法是 ``draw``。考虑以下三角形图表的示例：

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import DiagramGrid, XypicDiagramDrawer
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})

    要绘制此图表，其对象需要使用 :class:`DiagramGrid` 进行布局：

    >>> grid = DiagramGrid(diagram)

    最后，进行绘制：

    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    更多细节请参阅此方法的文档字符串。

    控制箭头外观可以使用格式化器。字典 ``arrow_formatters`` 将态射映射到格式化函数。格式化器接受一个 :class:`ArrowStringDescription` 并允许修改其公开的任何箭头属性。例如，要将所有具有属性 ``unique`` 的态射显示为虚线箭头，并在它们的名称前加上 `\exists !`，可以这样做：

    >>> def formatter(astr):
    ...   astr.label = r"\exists !" + astr.label
    ...   astr.arrow_style = "{-->}"
    >>> drawer.arrow_formatters["unique"] = formatter
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar@{-->}[d]_{\exists !g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    要修改图表中所有箭头的外观，设置 ``default_arrow_formatter``。例如，将所有态射标签稍微移动到箭头头部之外，使它们看起来更居中，操作如下：

    >>> def default_formatter(astr):
    ...   astr.label_displacement = "(0.45)"
    >>> drawer.default_arrow_formatter = default_formatter
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar@{-->}[d]_(0.45){\exists !g\circ f} \ar[r]^(0.45){f} & B \ar[ld]^(0.45){g} \\
    C &
    }

    在某些图表中，一些态射可能作为曲线箭头绘制。考虑以下图表：

    >>> D = Object("D")
    >>> E = Object("E")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])
    >>> grid = DiagramGrid(diagram)
    >>> drawer = XypicDiagramDrawer()
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_3mm/[ll]_{h} \\
    & C &
    }

    要控制默认曲线箭头的曲率程度，可以使用 ``unit`` 和 ``default_curving_amount`` 属性：

    >>> drawer.unit = "cm"
    >>> drawer.default_curving_amount = 1
    """

    arrow_formatters = {}  # 箭头格式化器，将态射属性映射到格式化函数
    default_arrow_formatter = None  # 默认箭头格式化器函数

    def draw(self, diagram, grid):
        """
        绘制给定图表和网格的 Xy-pic 表示。

        Args:
            diagram (:class:`~.Diagram`): 要绘制的图表对象。
            grid (:class:`DiagramGrid`): 用于布局图表对象的网格。

        Returns:
            str: Xy-pic 表示的字符串。
        """
        pass  # 实际的绘制逻辑在这里实现，这里只是占位符
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_1cm/[ll]_{h} \\
    & C &
    }

这部分是一个示例代码片段，展示了如何使用 `drawer` 对象来绘制一个简单的数学图表，并打印出结果。


    In some diagrams, there are multiple curved morphisms between the
    same two objects.  To control by how much the curving changes
    between two such successive morphisms, use
    ``default_curving_step``:

这是一个注释段落，解释了在某些图表中可能会出现多个曲线形态的态射连接同一对对象之间的情况。还说明了如何使用 `default_curving_step` 控制连续两个这样的态射之间曲线形态的变化量。


    >>> drawer.default_curving_step = 1
    >>> h1 = NamedMorphism(A, D, "h1")
    >>> diagram = Diagram([f, g, h, k, h1])
    >>> grid = DiagramGrid(diagram)
    >>> print(drawer.draw(diagram, grid))
    \xymatrix{
    A \ar[r]_{f} \ar@/^1cm/[rr]^{h_{1}} & B \ar[d]^{g} & D \ar[l]^{k} \ar@/_2cm/[ll]_{h} \\
    & C &
    }

这部分展示了如何修改 `drawer` 对象的 `default_curving_step` 属性，并使用新创建的 `NamedMorphism` 对象 `h1` 更新图表 `diagram` 和 `grid`，然后再次绘制图表以显示曲线形态的变化。


    The default value of ``default_curving_step`` is 4 units.

    See Also
    ========

    draw, ArrowStringDescription
    """

这是一段注释，指出了 `default_curving_step` 的默认值是 4 单位，并提供了相关的参考信息，包括 `draw` 和 `ArrowStringDescription` 相关内容。


    def __init__(self):

这是一个类的构造函数的定义，初始化了几个属性。


        self.unit = "mm"
        self.default_curving_amount = 3
        self.default_curving_step = 4

在构造函数中初始化了几个属性：`unit` 设置为 "mm"，`default_curving_amount` 设置为 3，`default_curving_step` 设置为 4。


        # This dictionary maps properties to the corresponding arrow
        # formatters.
        self.arrow_formatters = {}

创建了一个空字典 `arrow_formatters`，用于将属性映射到相应的箭头格式化器。


        # This is the default arrow formatter which will be applied to
        # each arrow independently of its properties.
        self.default_arrow_formatter = None

初始化了 `default_arrow_formatter` 属性为 `None`，表示每个箭头的默认格式化器为空。


    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    def _check_free_space_horizontal(dom_i, dom_j, cod_j, grid):

定义了一个静态方法 `_check_free_space_horizontal`，用于检查水平态射的自由空间，即在态射上方或下方是否有空闲的位置未被其他对象占据。


        """
        For a horizontal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """

这是 `_check_free_space_horizontal` 方法的文档字符串，说明了这个方法的作用是检查水平态射的自由空间。


        if dom_j < cod_j:
            (start, end) = (dom_j, cod_j)
            backwards = False
        else:
            (start, end) = (cod_j, dom_j)
            backwards = True

根据给定的起始对象和结束对象的坐标位置，确定要检查的范围，并设置 `backwards` 变量来表示是否是反向检查。


        # Check for free space above.
        if dom_i == 0:
            free_up = True
        else:
            free_up = all(grid[dom_i - 1, j] for j in
                          range(start, end + 1))

检查在当前态射的上方是否有空闲空间，如果当前对象位于最顶端，则认为有空闲空间，否则检查上方所有位置是否已被占据。


        # Check for free space below.
        if dom_i == grid.height - 1:
            free_down = True
        else:
            free_down = not any(grid[dom_i + 1, j] for j in
                                range(start, end + 1))

检查在当前态射的下方是否有空闲空间，如果当前对象位于最底端，则认为有空闲空间，否则检查下方所有位置是否已被占据。


        return (free_up, free_down, backwards)

返回一个元组，包含三个布尔值，分别表示上方是否有空闲空间、下方是否有空闲空间以及是否反向检查。


    @staticmethod

定义了另一个静态方法。
    # 定义一个静态方法，用于检查垂直态射的自由空间
    def _check_free_space_vertical(dom_i, cod_i, dom_j, grid):
        """
        For a vertical morphism, checks whether there is free space
        (i.e., space not occupied by any objects) to the left of the
        morphism or to the right of it.
        垂直态射的情况下，检查是否存在自由空间（即未被任何对象占据的空间）
        在态射的左侧或右侧。
        """
        # 根据态射起点和终点确定扫描方向和范围
        if dom_i < cod_i:
            (start, end) = (dom_i, cod_i)
            backwards = False
        else:
            (start, end) = (cod_i, dom_i)
            backwards = True

        # 检查左侧是否有空间
        if dom_j == 0:
            free_left = True
        else:
            # 如果不是第一列，检查从起点到终点的每一行在dom_j-1列是否有对象存在
            free_left = not any(grid[i, dom_j - 1] for i in range(start, end + 1))

        # 检查右侧是否有空间
        if dom_j == grid.width - 1:
            free_right = True
        else:
            # 如果不是最后一列，检查从起点到终点的每一行在dom_j+1列是否有对象存在
            free_right = not any(grid[i, dom_j + 1] for i in range(start, end + 1))

        # 返回左侧是否自由、右侧是否自由以及扫描方向的元组
        return (free_left, free_right, backwards)

    @staticmethod
    def _push_labels_out(self, morphisms_str_info, grid, object_coords):
        """
        For all straight morphisms which form the visual boundary of
        the laid out diagram, puts their labels on their outer sides.
        """

        def set_label_position(free1, free2, pos1, pos2, backwards, m_str_info):
            """
            Given the information about room available to one side and
            to the other side of a morphism (`free1` and `free2`),
            sets the position of the morphism label in such a way that
            it is on the freer side. This latter operation involves
            choosing between `pos1` and `pos2`, taking `backwards`
            into consideration.

            This function does nothing if either both `free1 == True`
            and `free2 == True` or both `free1 == False` and `free2 == False`.
            In either case, choosing one side over the other presents no advantage.
            """
            if backwards:
                (pos1, pos2) = (pos2, pos1)

            if free1 and not free2:
                m_str_info.label_position = pos1
            elif free2 and not free1:
                m_str_info.label_position = pos2

        # Iterate through each morphism and its associated straight information
        for m, m_str_info in morphisms_str_info.items():
            if m_str_info.curving or m_str_info.forced_label_position:
                # Skip morphisms that are curved or have a predefined label position
                continue

            if m.domain == m.codomain:
                # Skip loop morphisms as their labels are handled differently
                continue

            # Fetch coordinates of domain and codomain objects
            (dom_i, dom_j) = object_coords[m.domain]
            (cod_i, cod_j) = object_coords[m.codomain]

            if dom_i == cod_i:
                # Horizontal morphism
                (free_up, free_down,
                 backwards) = XypicDiagramDrawer._check_free_space_horizontal(
                     dom_i, dom_j, cod_j, grid)

                set_label_position(free_up, free_down, "^", "_",
                                   backwards, m_str_info)
            elif dom_j == cod_j:
                # Vertical morphism
                (free_left, free_right,
                 backwards) = XypicDiagramDrawer._check_free_space_vertical(
                     dom_i, cod_i, dom_j, grid)

                set_label_position(free_left, free_right, "_", "^",
                                   backwards, m_str_info)
            else:
                # Diagonal morphism
                (free_up, free_down,
                 backwards) = XypicDiagramDrawer._check_free_space_diagonal(
                     dom_i, cod_i, dom_j, cod_j, grid)

                set_label_position(free_up, free_down, "^", "_",
                                   backwards, m_str_info)
    def _morphism_sort_key(morphism, object_coords):
        """
        提供一个排序键函数，使得相邻对象之间的水平或垂直态射首先出现，
        然后是更远对象之间的水平或垂直态射，最后是所有其他态射。
        """
        # 获取态射的起始对象的坐标
        (i, j) = object_coords[morphism.domain]
        # 获取态射的目标对象的坐标
        (target_i, target_j) = object_coords[morphism.codomain]

        # 如果态射的起始对象和目标对象相同，则为环态射
        if morphism.domain == morphism.codomain:
            # 环态射排在对角线态射之后，这样可以确定环的曲线方向。
            return (3, 0, default_sort_key(morphism))

        # 如果目标对象与起始对象在同一行，则按距离排序
        if target_i == i:
            return (1, abs(target_j - j), default_sort_key(morphism))

        # 如果目标对象与起始对象在同一列，则按距离排序
        if target_j == j:
            return (1, abs(target_i - i), default_sort_key(morphism))

        # 对角线态射
        return (2, 0, default_sort_key(morphism))


    @staticmethod
    def _build_xypic_string(diagram, grid, morphisms,
                            morphisms_str_info, diagram_format):
        """
        给定一个描述图表态射的 :class:`ArrowStringDescription` 集合，
        以及图表对象布局信息，生成最终的 Xy-pic 图片。
        """
        # 构建对象与以其为定义域的态射之间的映射关系
        object_morphisms = {}
        for obj in diagram.objects:
            object_morphisms[obj] = []
        for morphism in morphisms:
            object_morphisms[morphism.domain].append(morphism)

        # 初始化结果字符串，开始 Xy-pic 图片的构建
        result = "\\xymatrix%s{\n" % diagram_format

        # 遍历网格的高度和宽度
        for i in range(grid.height):
            for j in range(grid.width):
                # 获取网格位置 (i, j) 处的对象
                obj = grid[i, j]
                if obj:
                    # 添加对象的 LaTeX 表示到结果字符串
                    result += latex(obj) + " "

                    # 获取以当前对象为定义域的待绘制态射
                    morphisms_to_draw = object_morphisms[obj]
                    for morphism in morphisms_to_draw:
                        # 添加态射的字符串信息到结果字符串
                        result += str(morphisms_str_info[morphism]) + " "

                # 在最后一列之前不添加 &
                if j < grid.width - 1:
                    result += "& "

            # 最后一行之前不添加换行符 \\
            if i < grid.height - 1:
                result += "\\\\"
            result += "\n"

        # 完成 Xy-pic 图片的构建
        result += "}\n"

        return result
# 定义函数 xypic_draw_diagram，用于绘制给定图表的 Xy-pic 表示
def xypic_draw_diagram(diagram, masked=None, diagram_format="",
                       groups=None, **hints):
    r"""
    提供一个快捷方式结合 :class:`DiagramGrid` 和 :class:`XypicDiagramDrawer`。
    返回 ``diagram`` 的 Xy-pic 表示。
    参数 ``masked`` 是一个不需要绘制的态射列表。
    参数 ``diagram_format`` 是插入 "\xymatrix" 后的格式字符串。
    ``groups`` 应该是一组逻辑分组。
    ``hints`` 将直接传递给 :class:`DiagramGrid` 的构造函数。

    关于参数的更多信息，请参见 :class:`DiagramGrid` 和 ``XypicDiagramDrawer.draw`` 的文档字符串。

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import xypic_draw_diagram
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g], {g * f: "unique"})
    >>> print(xypic_draw_diagram(diagram))
    \xymatrix{
    A \ar[d]_{g\circ f} \ar[r]^{f} & B \ar[ld]^{g} \\
    C &
    }

    See Also
    ========

    XypicDiagramDrawer, DiagramGrid
    """
    # 创建图表网格对象
    grid = DiagramGrid(diagram, groups, **hints)
    # 创建 XypicDiagramDrawer 对象
    drawer = XypicDiagramDrawer()
    # 调用 XypicDiagramDrawer 的 draw 方法绘制图表
    return drawer.draw(diagram, grid, masked, diagram_format)


# 定义函数 preview_diagram，结合了 xypic_draw_diagram 和 sympy.printing.preview 的功能
@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',))
def preview_diagram(diagram, masked=None, diagram_format="", groups=None,
                    output='png', viewer=None, euler=True, **hints):
    """
    结合了 ``xypic_draw_diagram`` 和 ``sympy.printing.preview`` 的功能。
    参数 ``masked``, ``diagram_format``, ``groups``, 和 ``hints`` 传递给 ``xypic_draw_diagram``，
    而 ``output``, ``viewer``, 和 ``euler`` 传递给 ``preview``。

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy.categories import preview_diagram
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g], {g * f: "unique"})
    >>> preview_diagram(d)

    See Also
    ========

    XypicDiagramDrawer
    """
    # 导入 preview 函数
    from sympy.printing import preview
    # 调用 xypic_draw_diagram 函数生成 LaTeX 输出
    latex_output = xypic_draw_diagram(diagram, masked, diagram_format,
                                      groups, **hints)
    # 调用 preview 函数预览生成的 LaTeX 输出
    preview(latex_output, output, viewer, euler, ("xypic",))
```