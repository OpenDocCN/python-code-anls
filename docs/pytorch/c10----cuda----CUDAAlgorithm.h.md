# `.\pytorch\c10\cuda\CUDAAlgorithm.h`

```py
#ifdef THRUST_DEVICE_LOWER_BOUND_WORKS
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#endif

// 在 c10::cuda 命名空间内定义 lower_bound 函数，根据是否定义了 THRUST_DEVICE_LOWER_BOUND_WORKS 来选择不同的实现方式
namespace c10::cuda {
#ifdef THRUST_DEVICE_LOWER_BOUND_WORKS
// 如果 THRUST_DEVICE_LOWER_BOUND_WORKS 被定义，则使用 Thrust 库提供的 lower_bound 函数进行实现
template <typename Iter, typename Scalar>
__forceinline__ __device__ Iter lower_bound(Iter start, Iter end, Scalar value) {
  return thrust::lower_bound(thrust::device, start, end, value);
}
#else
// 如果 THRUST_DEVICE_LOWER_BOUND_WORKS 未定义，则使用自定义的二分查找实现，因为 thrust::lower_bound 在设备上存在问题
// 参考自 https://github.com/pytorch/pytorch/blob/805120ab572efef66425c9f595d9c6c464383336/aten/src/ATen/native/cuda/Bucketization.cu#L28
template <typename Iter, typename Scalar>
__device__ Iter lower_bound(Iter start, Iter end, Scalar value) {
  // 使用二分查找的方式找到第一个大于等于 value 的位置
  while (start < end) {
    auto mid = start + ((end - start) >> 1);
    if (*mid < value) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return end;
}
#endif // THRUST_DEVICE_LOWER_BOUND_WORKS
} // namespace c10::cuda
```