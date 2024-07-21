# `.\pytorch\aten\src\ATen\native\sparse\cuda\StaticSort.h`

```
#pragma once
#include <cutlass/cutlass.h>

/**
 * A Functor class to create a sort for fixed sized arrays/containers with a
 * compile time generated Bose-Nelson sorting network.
 * \tparam NumElements  The number of elements in the array or container to
 * sort. \tparam T            The element type. \tparam Compare      A
 * comparator functor class that returns true if lhs < rhs.
 */
template <unsigned NumElements>
class StaticSort {
  /**
   * Internal struct to perform swapping of elements in an array or container.
   * \tparam A  Type of the array or container.
   */
  template <class A>
  struct Swap {
    /**
     * Swaps two elements in an array or container.
     * \tparam T  Type of elements.
     * \param v0  First element to swap.
     * \param v1  Second element to swap.
     */
    CUTLASS_HOST_DEVICE void s(T& v0, T& v1) {
      // Explicitly code out the Min and Max to nudge the compiler
      // to generate branchless code.
      T t = v0 < v1 ? v0 : v1; // Min
      v1 = v0 < v1 ? v1 : v0; // Max
      v0 = t;
    }

    /**
     * Constructor performing swap operation on specified indices.
     * \param a    Reference to the array or container.
     * \param i0   Index of the first element.
     * \param i1   Index of the second element.
     */
    CUTLASS_HOST_DEVICE Swap(A& a, const int& i0, const int& i1) {
      s(a[i0], a[i1]);
    }
  };

  /**
   * Internal struct implementing the Bose-Nelson sorting network.
   * \tparam A  Type of the array or container.
   * \tparam I  Starting index of the first element pair.
   * \tparam J  Starting index of the second element pair.
   * \tparam X  Number of elements to sort at this level.
   * \tparam Y  Number of comparisons to perform at this level.
   */
  template <class A, int I, int J, int X, int Y>
  struct PB {
    /**
     * Constructor implementing the sorting network recursively.
     * \param a  Reference to the array or container.
     */
    CUTLASS_HOST_DEVICE PB(A& a) {
      enum {
        L = X >> 1,
        M = (X & 1 ? Y : Y + 1) >> 1,
        IAddL = I + L,
        XSubL = X - L
      };
      PB<A, I, J, L, M> p0(a);
      PB<A, IAddL, J + M, XSubL, Y - M> p1(a);
      PB<A, IAddL, J, XSubL, M> p2(a);
    }
  };

  /**
   * Specialization for a 1x1 element to perform a swap.
   * \tparam A  Type of the array or container.
   * \tparam I  Index of the first element.
   * \tparam J  Index of the second element.
   */
  template <class A, int I, int J>
  struct PB<A, I, J, 1, 1> {
    /**
     * Constructor performing a swap between two elements.
     * \param a  Reference to the array or container.
     */
    CUTLASS_HOST_DEVICE PB(A& a) {
      Swap<A> s(a, I - 1, J - 1);
    }
  };

  /**
   * Specialization for a 1x2 element to perform two swaps.
   * \tparam A  Type of the array or container.
   * \tparam I  Index of the first element.
   * \tparam J  Index of the second element.
   */
  template <class A, int I, int J>
  struct PB<A, I, J, 1, 2> {
    /**
     * Constructor performing two swaps between elements.
     * \param a  Reference to the array or container.
     */
    CUTLASS_HOST_DEVICE PB(A& a) {
      Swap<A> s0(a, I - 1, J);
      Swap<A> s1(a, I - 1, J - 1);
    }
  };

  /**
   * Specialization for a 2x1 element to perform two swaps.
   * \tparam A  Type of the array or container.
   * \tparam I  Index of the first element.
   * \tparam J  Index of the second element.
   */
  template <class A, int I, int J>
  struct PB<A, I, J, 2, 1> {
    /**
     * Constructor performing two swaps between elements.
     * \param a  Reference to the array or container.
     */
    CUTLASS_HOST_DEVICE PB(A& a) {
      Swap<A> s0(a, I - 1, J - 1);
      Swap<A> s1(a, I, J - 1);
    }
  };

  /**
   * Internal struct implementing the final pass of the sorting network.
   * \tparam A     Type of the array or container.
   * \tparam I     Starting index of the first element pair.
   * \tparam M     Number of elements to sort at this level.
   * \tparam Stop  Indicates whether further partitioning is needed.
   */
  template <class A, int I, int M, bool Stop = false>
  struct PS {
    /**
     * Constructor implementing the final pass of the sorting network.
     * \param a  Reference to the array or container.
     */
    CUTLASS_HOST_DEVICE PS(A& a) {
      enum { L = M >> 1, IAddL = I + L, MSubL = M - L };
      PS<A, I, L, (L <= 1)> ps0(a);
      PS<A, IAddL, MSubL, (MSubL <= 1)> ps1(a);
      PB<A, I, IAddL, L, MSubL> pb(a);
    }
  };

  /**
   * Specialization for stopping further partitioning.
   * \tparam A  Type of the array or container.
   * \tparam I  Starting index of the first element pair.
   * \tparam M  Number of elements to sort at this level.
   */
  template <class A, int I, int M>
  struct PS<A, I, M, true> {
    /**
     * Constructor for the case where further partitioning is not needed.
     * \param a  Reference to the array or container.
     */
    CUTLASS_HOST_DEVICE PS(A& a) {}
  };

 public:
  /**
   * Sorts the array/container arr using the Bose-Nelson sorting network.
   * \param  arr  The array/container to be sorted.
   */
  template <class Container>
  CUTLASS_HOST_DEVICE void operator()(Container& arr) const {
    PS<Container, 1, NumElements, (NumElements <= 1)> ps(arr);
  };

  /**
   * Sorts the array arr using the Bose-Nelson sorting network.
   * \param  arr  The array to be sorted.
   */
  template <class T>
  CUTLASS_HOST_DEVICE void operator()(T* arr) const {
    PS<T*, 1, NumElements, (NumElements <= 1)> ps(arr);
  };
};
```