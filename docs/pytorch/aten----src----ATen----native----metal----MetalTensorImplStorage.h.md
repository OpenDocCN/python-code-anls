# `.\pytorch\aten\src\ATen\native\metal\MetalTensorImplStorage.h`

```
namespace at::native::metal {

// 进入 at::native::metal 命名空间


class MPSImageWrapper;
class MetalTensorImplStorage final {
  class Impl;

 public:
  MetalTensorImplStorage(){};

// MetalTensorImplStorage 类的声明，使用 final 关键字表示不能被继承，没有参数的默认构造函数的定义


  MetalTensorImplStorage(const std::vector<int64_t>& sizes);

// MetalTensorImplStorage 类的构造函数声明，接受 sizes 参数的引用


  MetalTensorImplStorage(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides);

// MetalTensorImplStorage 类的构造函数声明，接受 sizes 和 strides 两个参数的引用


  ~MetalTensorImplStorage() = default;

// MetalTensorImplStorage 类的析构函数声明，使用默认实现


  MetalTensorImplStorage(MetalTensorImplStorage&&) = default;
  MetalTensorImplStorage& operator=(MetalTensorImplStorage&&) = default;

// 移动构造函数和移动赋值运算符的声明，都使用默认实现


  MetalTensorImplStorage(const MetalTensorImplStorage&) = default;
  MetalTensorImplStorage& operator=(const MetalTensorImplStorage&) = default;

// 拷贝构造函数和拷贝赋值运算符的声明，都使用默认实现


  friend std::ostream& operator<<(
      std::ostream& output,
      const MetalTensorImplStorage& mt);

// 声明友元函数 operator<<，用于将 MetalTensorImplStorage 对象输出到 ostream


  bool defined() const;
  IntArrayRef sizes() const;
  IntArrayRef strides() const;
  int64_t dim() const;
  int64_t numel() const;
  void set_data_from_host(const float* inputData);
  void copy_data_to_host(float* host);
  MPSImageWrapper* texture() const;

// 公共成员函数的声明，分别用于判断是否定义、获取 sizes 和 strides、获取维度数和元素数量、从主机设置数据、从主机复制数据、获取 MPSImageWrapper 指针


 private:
  std::shared_ptr<Impl> impl();
  std::shared_ptr<const Impl> impl() const;
  std::shared_ptr<Impl> _impl;
};

// 私有成员函数 impl() 的声明，返回 Impl 类的 shared_ptr，常量和非常量版本，以及 _impl 的声明为 MetalTensorImplStorage 类的私有成员变量

} // namespace at::native::metal

// 结束 at::native::metal 命名空间
```