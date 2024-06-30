# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\_partition_nodes.pyx`

```
# BinaryTrees rely on partial sorts to partition their nodes during their
# initialisation.
#
# The C++ std library exposes nth_element, an efficient partial sort for this
# situation which has a linear time complexity as well as the best performances.
#
# To use std::algorithm::nth_element, a few fixture are defined using Cython:
# - partition_node_indices, a Cython function used in BinaryTrees, that calls
# - partition_node_indices_inner, a C++ function that wraps nth_element and uses
# - an IndexComparator to state how to compare KDTrees' indices
#
# IndexComparator has been defined so that partial sorts are stable with
# respect to the nodes initial indices.
#
# See for reference:
#  - https://en.cppreference.com/w/cpp/algorithm/nth_element.
#  - https://github.com/scikit-learn/scikit-learn/pull/11103
#  - https://github.com/scikit-learn/scikit-learn/pull/19473
from cython cimport floating


cdef extern from *:
    """
    #include <algorithm>

    template<class D, class I>
    class IndexComparator {
    private:
        const D *data;
        I split_dim, n_features;
    public:
        IndexComparator(const D *data, const I &split_dim, const I &n_features):
            data(data), split_dim(split_dim), n_features(n_features) {}

        // Comparison operator for IndexComparator class, compares elements based on split dimension
        bool operator()(const I &a, const I &b) const {
            D a_value = data[a * n_features + split_dim];
            D b_value = data[b * n_features + split_dim];
            return a_value == b_value ? a < b : a_value < b_value;
        }
    };

    // Function template to partition node indices using nth_element
    template<class D, class I>
    void partition_node_indices_inner(
        const D *data,
        I *node_indices,
        const I &split_dim,
        const I &split_index,
        const I &n_features,
        const I &n_points) {
        // Create an instance of IndexComparator
        IndexComparator<D, I> index_comparator(data, split_dim, n_features);
        // Call std::nth_element to partially sort node_indices
        std::nth_element(
            node_indices,
            node_indices + split_index,
            node_indices + n_points,
            index_comparator);
    }
    """
    // Declaration of partition_node_indices_inner function using Cython
    void partition_node_indices_inner[D, I](
                const D *data,
                I *node_indices,
                I split_dim,
                I split_index,
                I n_features,
                I n_points) except +


cdef int partition_node_indices(
        const floating *data,
        intp_t *node_indices,
        intp_t split_dim,
        intp_t split_index,
        intp_t n_features,
        intp_t n_points) except -1:
    """Partition points in the node into two equal-sized groups.

    Upon return, the values in node_indices will be rearranged such that
    (assuming numpy-style indexing):

        data[node_indices[0:split_index], split_dim]
          <= data[node_indices[split_index], split_dim]

    and

        data[node_indices[split_index], split_dim]
          <= data[node_indices[split_index:n_points], split_dim]

    The algorithm is essentially a partial in-place quicksort around a
    set pivot.

    Parameters
    ----------
    data : const floating *
        Pointer to the data array.
    node_indices : intp_t *
        Pointer to the array of node indices.
    split_dim : intp_t
        Dimension along which to split the nodes.
    split_index : intp_t
        Index of the split.
    n_features : intp_t
        Number of features in the data.
    n_points : intp_t
        Number of points (or nodes) to partition.

    Returns
    -------
    int
        Returns -1 on error (currently not used).

    """
    data : double pointer
        指向训练数据的二维数组的指针，形状为[N, n_features]。
        N 必须大于 node_indices 中的任何值。
    node_indices : int pointer
        指向长度为 n_points 的一维数组的指针。这列出了当前节点中每个点的索引。这将被就地修改。
    split_dim : int
        要进行分割的维度。通常通过 find_node_split_dim 程序计算得出。
    split_index : int
        在围绕其分割点的 node_indices 中的索引。
    n_features: int
        data 指针指向的二维数组中的特征数（即列数）。
    n_points : int
        node_indices 的长度。这也是原始数据集中的点数。
    Returns
    -------
    status : int
        整数退出状态。返回时，node_indices 的内容将根据上述说明进行修改。
    """
    partition_node_indices_inner(
        data,
        node_indices,
        split_dim,
        split_index,
        n_features,
        n_points)
    返回 0
```