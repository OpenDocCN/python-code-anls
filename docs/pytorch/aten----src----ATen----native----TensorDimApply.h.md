# `.\pytorch\aten\src\ATen\native\TensorDimApply.h`

```py
#pragma once
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

namespace at::native {

// tensor_dim_apply3函数的实现，用于在指定维度上应用函数func
// T1和T2是模板类型参数，Function是函数对象类型参数
template<typename T1, typename T2, typename Function>
void tensor_dim_apply3(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim, Function func) {
  
  // 获取输入张量的维度数
  int ndims = self.dim();
  
  // 标志是否完成了tensor_dim_apply的标志，初始为0
  int tensor_dim_apply_has_finished = 0;
  
  // 计数器，记录当前遍历到的各维度的索引
  std::vector<int64_t> counter(ndims, 0);
  
  // 获取输入张量self的数据指针，类型为T1
  const T1* self_data = self.const_data_ptr<T1>();
  
  // 获取输出张量values的数据指针，类型为T1
  T1* values_data = values.data_ptr<T1>();
  
  // 获取输出张量indices的数据指针，类型为T2
  T2* indices_data = indices.data_ptr<T2>();
  
  // 获取输入张量self在指定维度dim上的步长
  int64_t self_stride = self.stride(dim);
  
  // 获取输出张量values在指定维度dim上的步长
  int64_t values_stride = values.stride(dim);
  
  // 获取输出张量indices在指定维度dim上的步长
  int64_t indices_stride = indices.stride(dim);
  
  // 获取输入张量self在指定维度dim上的大小
  int self_dim_size = self.size(dim);

  // 循环执行tensor_dim_apply操作，直到所有维度上的元素都被处理
  while (!tensor_dim_apply_has_finished) {
    
    // 调用传入的函数对象func，对当前位置的数据进行处理
    func(self_data, values_data, indices_data, self_dim_size, self_stride, values_stride, indices_stride);
    
    // 如果张量只有1维，则直接退出循环
    if (ndims == 1) {
       break;
    }
    
    // 遍历所有维度
    for (const auto dim_i : c10::irange(ndims)) {
      
      // 如果当前维度是指定的dim维度
      if (dim_i == dim) {
        
        // 如果当前维度是最后一个维度，则标记tensor_dim_apply操作完成，并跳出循环
        if (dim_i == (ndims - 1)) {
          tensor_dim_apply_has_finished = 1;
          break;
        }
        
        // 否则继续下一个维度的处理
        continue;
      }
      
      // 对当前维度的计数器加1，移动数据指针到下一个位置
      counter[dim_i]++;
      self_data += self.stride(dim_i);
      values_data += values.stride(dim_i);
      indices_data += indices.stride(dim_i);

      // 如果当前维度的计数器达到了其大小
      if (counter[dim_i] == self.size(dim_i)) {
        
        // 如果当前维度是最后一个维度，则标记tensor_dim_apply操作完成，并跳出循环
        if (dim_i == ndims-1) {
          tensor_dim_apply_has_finished = 1;
          break;
        } else {
          // 否则回退当前维度的指针和计数器
          self_data -= counter[dim_i]*self.stride(dim_i);
          values_data -= counter[dim_i]*values.stride(dim_i);
          indices_data -= counter[dim_i]*indices.stride(dim_i);
          counter[dim_i] = 0;
        }
      } else {
        // 如果计数器未达到大小，则终止当前循环，继续下一个位置处理
        break;
     }
    }
  }
}

} // namespace at::native


这段代码实现了一个函数 `tensor_dim_apply3`，用于在张量的指定维度上应用函数 `func`。函数中使用了模板类型参数 `T1` 和 `T2`，以及函数对象类型参数 `Function`，允许在不同类型的张量上执行类似操作。
```