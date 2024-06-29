# `.\numpy\tools\swig\test\Array2.cxx`

```
// 包含 Array2 类的声明文件
#include "Array2.h"
// 包含用于字符串流操作的头文件
#include <sstream>

// 默认构造函数
Array2::Array2() :
  _ownData(false), _nrows(0), _ncols(), _buffer(0), _rows(0)
{ }

// 大小和数组构造函数
Array2::Array2(int nrows, int ncols, long* data) :
  _ownData(false), _nrows(0), _ncols(), _buffer(0), _rows(0)
{
  // 调整数组大小，并使用传入的数据进行初始化
  resize(nrows, ncols, data);
}

// 复制构造函数
Array2::Array2(const Array2 & source) :
  _nrows(source._nrows), _ncols(source._ncols)
{
  // 分配内存并将数据复制到当前对象
  _ownData = true;
  allocateMemory();
  *this = source;
}

// 析构函数
Array2::~Array2()
{
  // 释放对象占用的内存
  deallocateMemory();
}

// 赋值运算符重载
Array2 & Array2::operator=(const Array2 & source)
{
  // 按最小行列数复制数据到当前对象
  int nrows = _nrows < source._nrows ? _nrows : source._nrows;
  int ncols = _ncols < source._ncols ? _ncols : source._ncols;
  for (int i=0; i < nrows; ++i)
  {
    for (int j=0; j < ncols; ++j)
    {
      (*this)[i][j] = source[i][j];
    }
  }
  return *this;
}

// 等于运算符重载
bool Array2::operator==(const Array2 & other) const
{
  // 检查是否行数、列数及所有元素相同
  if (_nrows != other._nrows) return false;
  if (_ncols != other._ncols) return false;
  for (int i=0; i < _nrows; ++i)
  {
    for (int j=0; j < _ncols; ++j)
    {
      if ((*this)[i][j] != other[i][j]) return false;
    }
  }
  return true;
}

// 获取行数
int Array2::nrows() const
{
  return _nrows;
}

// 获取列数
int Array2::ncols() const
{
  return _ncols;
}

// 调整数组大小
void Array2::resize(int nrows, int ncols, long* data)
{
  // 检查行数和列数是否合法，如果相同则直接返回，否则重新分配内存
  if (nrows < 0) throw std::invalid_argument("Array2 nrows less than 0");
  if (ncols < 0) throw std::invalid_argument("Array2 ncols less than 0");
  if (nrows == _nrows && ncols == _ncols) return;
  deallocateMemory();
  _nrows = nrows;
  _ncols = ncols;
  if (!data)
  {
    allocateMemory();
  }
  else
  {
    _ownData = false;
    _buffer  = data;
    allocateRows();
  }
}

// 仅调整数组大小（重载版本）
void Array2::resize(int nrows, int ncols)
{
  resize(nrows, ncols, nullptr);
}

// 设置元素访问器
Array1 & Array2::operator[](int i)
{
  // 检查行索引是否在合法范围内，然后返回对应行对象
  if (i < 0 || i >= _nrows) throw std::out_of_range("Array2 row index out of range");
  return _rows[i];
}

// 获取元素访问器
const Array1 & Array2::operator[](int i) const
{
  // 检查行索引是否在合法范围内，然后返回对应行对象（常量版本）
  if (i < 0 || i >= _nrows) throw std::out_of_range("Array2 row index out of range");
  return _rows[i];
}

// 返回数组的字符串表示
std::string Array2::asString() const
{
  std::stringstream result;
  result << "[ ";
  for (int i=0; i < _nrows; ++i)
  {
    if (i > 0) result << "  ";
    result << (*this)[i].asString();
    if (i < _nrows-1) result << "," << std::endl;
  }
  result << " ]" << std::endl;
  return result.str();
}

// 获取视图
void Array2::view(int* nrows, int* ncols, long** data) const
{
  // 返回当前数组的行数、列数及数据指针
  *nrows = _nrows;
  *ncols = _ncols;
  *data  = _buffer;
}

// 私有方法：分配内存
void Array2::allocateMemory()
{
  // 如果行列数为零，则将数据标记为非拥有，并清空缓冲区及行对象
  if (_nrows * _ncols == 0)
  {
    _ownData = false;
    _buffer  = 0;
    _rows    = 0;
  }
  else
  {
    // 否则分配内存，并标记为拥有数据，然后分配每一行的内存
    _ownData = true;
    _buffer = new long[_nrows*_ncols];
    allocateRows();
  }
}

// 私有方法：分配每一行的内存
void Array2::allocateRows()
{
  _rows = new Array1[_nrows];
  for (int i=0; i < _nrows; ++i)
  {
    _rows[i].resize(_ncols, &_buffer[i*_ncols]);



// 调整第 i 行的大小，确保其有 _ncols 列，使用 _buffer 中的数据作为初始内容
_rows[i].resize(_ncols, &_buffer[i*_ncols]);


这行代码的作用是调整 `_rows` 中第 `i` 行的大小，确保它包含 `_ncols` 列，并使用 `_buffer` 中第 `i*_ncols` 列开始的数据作为初始内容。`.resize()` 方法用于调整容器大小，并可以指定初始值。
}

void Array2::deallocateMemory()
{
  // 检查是否需要释放内存：确保_ownData为true，且_nrows和_ncols大于0，_buffer非空
  if (_ownData && _nrows*_ncols && _buffer)
  {
    // 删除_rows数组，释放行指针内存
    delete [] _rows;
    // 删除_buffer数组，释放数据内存
    delete [] _buffer;
  }
  // 重置所有成员变量，表示内存已被释放
  _ownData = false;
  _nrows   = 0;
  _ncols   = 0;
  _buffer  = 0;
  _rows    = 0;
}
```