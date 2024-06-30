# `D:\src\scipysrc\scikit-learn\sklearn\tree\_reingold_tilford.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

# 定义一个绘制树形结构的类
class DrawTree:
    # 初始化方法，接受树结构、父节点、深度和节点编号等参数
    def __init__(self, tree, parent=None, depth=0, number=1):
        # 初始化节点的 x 和 y 坐标，x 初值设为 -1.0，y 为节点的深度
        self.x = -1.0
        self.y = depth
        # 存储树结构和子节点列表，递归创建子节点
        self.tree = tree
        self.children = [
            DrawTree(c, self, depth + 1, i + 1) for i, c in enumerate(tree.children)
        ]
        # 记录父节点、线程、模数、祖先、变化和偏移等属性
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        # 节点在兄弟节点中的编号，从 1 开始
        self.number = number

    # 返回左侧子节点或者线程，如果没有子节点则返回 None
    def left(self):
        return self.thread or len(self.children) and self.children[0]

    # 返回右侧子节点或者线程，如果没有子节点则返回 None
    def right(self):
        return self.thread or len(self.children) and self.children[-1]

    # 返回左侧兄弟节点，如果没有则返回 None
    def lbrother(self):
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return n
                else:
                    n = node
        return n

    # 获取最左侧的兄弟节点
    def get_lmost_sibling(self):
        if not self._lmost_sibling and self.parent and self != self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    # 属性：最左侧的兄弟节点
    lmost_sibling = property(get_lmost_sibling)

    # 返回节点的字符串表示，包括树结构、x 坐标和模数
    def __str__(self):
        return "%s: x=%s mod=%s" % (self.tree, self.x, self.mod)

    # 返回节点的字符串表示，调用 __str__() 方法
    def __repr__(self):
        return self.__str__()

    # 计算节点及其子节点的最大范围
    def max_extents(self):
        extents = [c.max_extents() for c in self.children]
        extents.append((self.x, self.y))
        return np.max(extents, axis=0)


# Buchheim 算法的主函数，返回计算后的树结构
def buchheim(tree):
    dt = first_walk(DrawTree(tree))
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt


# 第三次遍历树结构，将节点及其子节点的 x 坐标增加 n
def third_walk(tree, n):
    tree.x += n
    for c in tree.children:
        third_walk(c, n)


# 第一次遍历树结构，计算节点的 x 坐标并调整它们的位置
def first_walk(v, distance=1.0):
    if len(v.children) == 0:
        if v.lmost_sibling:
            v.x = v.lbrother().x + distance
        else:
            v.x = 0.0
    else:
        default_ancestor = v.children[0]
        for w in v.children:
            first_walk(w)
            default_ancestor = apportion(w, default_ancestor, distance)
        execute_shifts(v)

        midpoint = (v.children[0].x + v.children[-1].x) / 2

        w = v.lbrother()
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint
    return v


# 分配节点的位置，确保节点之间的间距符合要求
def apportion(v, default_ancestor, distance):
    w = v.lbrother()
    if w is not None:
        # 如果 w 不为 None，则执行以下操作
        # 根据布赫海姆表示法：
        # i == 内侧; o == 外侧; r == 右侧; l == 左侧; r = +; l = -
        
        # 初始化节点的各种变量
        vir = vor = v
        vil = w
        vol = v.lmost_sibling
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        
        # 当内侧节点的右侧和外侧节点的左侧都存在时，执行以下循环
        while vil.right() and vir.left():
            # 移动内侧和外侧节点及其子树，保持它们的相对位置
            vil = vil.right()
            vir = vir.left()
            vol = vol.left()
            vor = vor.right()
            
            # 将外侧节点的祖先设置为当前节点 v
            vor.ancestor = v
            
            # 计算需要移动的距离
            shift = (vil.x + sil) - (vir.x + sir) + distance
            
            # 如果需要移动的距离大于 0，则移动子树
            if shift > 0:
                move_subtree(ancestor(vil, v, default_ancestor), v, shift)
                sir = sir + shift
                sor = sor + shift
            
            # 更新各节点的 mod 值
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        
        # 如果内侧节点有右侧节点而外侧节点没有右侧节点
        if vil.right() and not vor.right():
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            # 如果外侧节点有左侧节点而内侧节点没有左侧节点
            if vir.left() and not vol.left():
                vol.thread = vir.left()
                vol.mod += sir - sol
            
            # 默认祖先节点设为当前节点 v
            default_ancestor = v
    
    # 返回默认祖先节点
    return default_ancestor
# 定义一个函数 move_subtree，用于移动子树在树结构中的位置
def move_subtree(wl, wr, shift):
    # 计算需要移动的子树数量
    subtrees = wr.number - wl.number
    # 更新右子树 wr 的位置和调整值
    wr.change -= shift / subtrees
    wr.shift += shift
    # 更新左子树 wl 的调整值
    wl.change += shift / subtrees
    # 更新右子树 wr 的 x 坐标和 mod 值
    wr.x += shift
    wr.mod += shift


# 定义一个函数 execute_shifts，用于执行树中节点的位置调整
def execute_shifts(v):
    shift = change = 0
    # 遍历 v 的子节点，从后向前
    for w in v.children[::-1]:
        # 更新节点 w 的 x 坐标和 mod 值
        w.x += shift
        w.mod += shift
        # 累加节点 w 的 change 值到总变化量 change
        change += w.change
        # 累加节点 w 的 shift 和 change 到总位移量 shift
        shift += w.shift + change


# 定义一个函数 ancestor，用于查找节点 v 的祖先节点
def ancestor(vil, v, default_ancestor):
    # 如果 vil 的祖先在 v 的父节点的子节点中，则返回 vil 的祖先节点
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    else:
        # 否则返回默认的祖先节点
        return default_ancestor


# 定义一个函数 second_walk，进行第二次遍历树结构，并调整节点的位置
def second_walk(v, m=0, depth=0, min=None):
    # 更新节点 v 的 x 坐标和 y 坐标
    v.x += m
    v.y = depth

    # 更新最小 x 坐标值 min
    if min is None or v.x < min:
        min = v.x

    # 递归遍历节点 v 的子节点，并更新最小 x 坐标值
    for w in v.children:
        min = second_walk(w, m + v.mod, depth + 1, min)

    return min


# 定义一个树结构的类 Tree
class Tree:
    def __init__(self, label="", node_id=-1, *children):
        self.label = label
        self.node_id = node_id
        if children:
            self.children = children
        else:
            self.children = []
```