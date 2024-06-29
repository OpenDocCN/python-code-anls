# `.\numpy\tools\swig\test\Farray.cxx`

```
// 包含自定义头文件 "Farray.h" 和标准头文件 <sstream>
#include "Farray.h"
#include <sstream>

// Farray 类的 Size 构造函数
Farray::Farray(int nrows, int ncols) :
  _nrows(nrows), _ncols(ncols), _buffer(0)
{
  // 分配内存空间
  allocateMemory();
}

// Farray 类的复制构造函数
Farray::Farray(const Farray & source) :
  _nrows(source._nrows), _ncols(source._ncols)
{
  // 分配内存空间
  allocateMemory();
  // 使用赋值运算符进行复制
  *this = source;
}

// Farray 类的析构函数
Farray::~Farray()
{
  // 释放动态分配的内存
  delete [] _buffer;
}

// Farray 类的赋值运算符重载
Farray & Farray::operator=(const Farray & source)
{
  // 确定有效的行和列数
  int nrows = _nrows < source._nrows ? _nrows : source._nrows;
  int ncols = _ncols < source._ncols ? _ncols : source._ncols;
  // 逐个元素进行赋值
  for (int i=0; i < nrows; ++i)
  {
    for (int j=0; j < ncols; ++j)
    {
      (*this)(i,j) = source(i,j);
    }
  }
  return *this;
}

// Farray 类的相等运算符重载
bool Farray::operator==(const Farray & other) const
{
  // 检查行数和列数是否相等
  if (_nrows != other._nrows) return false;
  if (_ncols != other._ncols) return false;
  // 逐个元素比较
  for (int i=0; i < _nrows; ++i)
  {
    for (int j=0; j < _ncols; ++j)
    {
      if ((*this)(i,j) != other(i,j)) return false;
    }
  }
  return true;
}

// 获取行数的访问器
int Farray::nrows() const
{
  return _nrows;
}

// 获取列数的访问器
int Farray::ncols() const
{
  return _ncols;
}

// 设置元素的访问器
long & Farray::operator()(int i, int j)
{
  // 检查行和列的索引是否有效
  if (i < 0 || i >= _nrows) throw std::out_of_range("Farray row index out of range");
  if (j < 0 || j >= _ncols) throw std::out_of_range("Farray col index out of range");
  return _buffer[offset(i,j)];
}

// 获取元素的访问器
const long & Farray::operator()(int i, int j) const
{
  // 检查行和列的索引是否有效
  if (i < 0 || i >= _nrows) throw std::out_of_range("Farray row index out of range");
  if (j < 0 || j >= _ncols) throw std::out_of_range("Farray col index out of range");
  return _buffer[offset(i,j)];
}

// 以字符串形式输出 Farray 对象
std::string Farray::asString() const
{
  std::stringstream result;
  result << "[ ";
  // 遍历数组元素并格式化输出
  for (int i=0; i < _nrows; ++i)
  {
    if (i > 0) result << "  ";
    result << "[";
    for (int j=0; j < _ncols; ++j)
    {
      result << " " << (*this)(i,j);
      if (j < _ncols-1) result << ",";
    }
    result << " ]";
    if (i < _nrows-1) result << "," << std::endl;
  }
  result << " ]" << std::endl;
  return result.str();
}

// 获取视图的访问器
void Farray::view(int* nrows, int* ncols, long** data) const
{
  // 返回行数、列数和数据指针的引用
  *nrows = _nrows;
  *ncols = _ncols;
  *data  = _buffer;
}

// 私有方法：分配内存空间
void Farray::allocateMemory()
{
  // 检查行数和列数是否有效
  if (_nrows <= 0) throw std::invalid_argument("Farray nrows <= 0");
  if (_ncols <= 0) throw std::invalid_argument("Farray ncols <= 0");
  // 分配内存空间
  _buffer = new long[_nrows*_ncols];
}

// 内联方法：计算元素在数组中的偏移量
inline int Farray::offset(int i, int j) const
{
  return i + j * _nrows;
}
```