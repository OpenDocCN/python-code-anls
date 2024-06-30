# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\dense.h`

```
#ifndef __DENSE_H__
#define __DENSE_H__

// Simplified BLAS routines and other dense linear algebra functions

/*
 * Level 1
 */

// y += a*x
// 向量加法：将向量 x 乘以标量 a 后加到向量 y 上
template <class I, class T>
void axpy(const I n, const T a, const T * x, T * y){
    for(I i = 0; i < n; i++){
        y[i] += a * x[i];
    }
}

// scale a vector in-place
// 向量缩放：将向量 x 中的每个元素乘以标量 a
template <class I, class T>
void scal(const I n, const T a, T * x){
    for(I i = 0; i < n; i++){
        x[i] *= a;
    }
}


// dot product
// 向量点积：计算向量 x 和 y 的点积
template <class I, class T>
T dot(const I n, const T * x, const T * y){
    T dp = 0;
    for(I i = 0; i < n; i++){
        dp += x[i] * y[i];
    }
    return dp;
}


// vectorize a binary operation
// 向量二元操作：对向量 x 和 y 中对应元素进行二元操作，并将结果存入向量 z
template<class I, class T, class binary_operator>
void vector_binop(const I n, const T * x, const T * y, T * z, 
                  const binary_operator& op)
{
    for(I i = 0; i < n; i++){
        z[i] = op(x[i],y[i]);
    }
}

//template<class I, class T>
//void vector_multiply(const I n, const T * x, const T * y, T * z){
//{
//    vector_binop(n,x,y,z, std::multiplies<T>() );
//}



// Level 2
// 矩阵向量乘法：将矩阵 A 与向量 x 相乘，结果存入向量 y
template <class I, class T>
void gemv(const I m, const I n, const T * A, const T * x, T * y){
    for(I i = 0; i < m; i++){
        T dot = y[i];
        for(I j = 0; j < n; j++){
            dot += A[(npy_intp)n * i + j] * x[j];
        }
        y[i] = dot;
    }
}

// Level 3
// 矩阵乘法：将矩阵 A 与矩阵 B 相乘，结果存入矩阵 C
template <class I, class T>
void gemm(const I m, const I n, const I k, const T * A, const T * B, T * C){
    for(I i = 0; i < m; i++){
        for(I j = 0; j < n; j++){
            T dot = C[(npy_intp)n * i + j];
            for(I _d = 0; _d < k; _d++){
                dot += A[(npy_intp)k * i + _d] * B[(npy_intp)n * _d + j];
            }
            C[(npy_intp)n * i + j] = dot;
        }
    }
}


#endif
```