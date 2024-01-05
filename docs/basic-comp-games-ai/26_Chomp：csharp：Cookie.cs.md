# `26_Chomp\csharp\Cookie.cs`

```
using System.Text;  // 导入 System.Text 命名空间，以便在代码中使用其中的类和方法

namespace Chomp;  // 声明 Chomp 命名空间，用于组织和管理代码

internal class Cookie  // 声明 Cookie 类，并指定为 internal 访问修饰符，表示只能在当前程序集中访问

{
    private readonly int _rowCount;  // 声明私有只读字段 _rowCount，用于存储行数
    private readonly int _columnCount;  // 声明私有只读字段 _columnCount，用于存储列数
    private readonly char[][] _bits;  // 声明私有只读字段 _bits，用于存储字符数组的二维数组

    public Cookie(int rowCount, int columnCount)  // 声明 Cookie 类的构造函数，接受行数和列数作为参数
    {
        _rowCount = rowCount;  // 将参数 rowCount 的值赋给 _rowCount 字段
        _columnCount = columnCount;  // 将参数 columnCount 的值赋给 _columnCount 字段

        // The calls to Math.Max here are to duplicate the original behaviour
        // when negative values are given for the row or column count.
        _bits = new char[Math.Max(_rowCount, 1)][];  // 使用 Math.Max 方法创建一个行数为 _rowCount 或 1 的字符数组
        for (int row = 0; row < _bits.Length; row++)  // 遍历 _bits 数组的每一行
        {
            _bits[row] = Enumerable.Repeat('*', Math.Max(_columnCount, 1)).ToArray();  # 在_bits数组的第row行中，创建一个长度为Math.Max(_columnCount, 1)的数组，数组元素为'*'
        }
        _bits[0][0] = 'P';  # 将_bits数组的第一行第一列的元素赋值为'P'
    }

    public bool TryChomp(int row, int column, out char chomped)
    {
        if (row < 1 || row > _rowCount || column < 1 || column > _columnCount || _bits[row - 1][column - 1] == ' ')  # 如果row或column小于1，或者大于对应的行数或列数，或者对应的元素为' '，则执行以下操作
        {
            chomped = default;  # 将chomped赋值为默认值
            return false;  # 返回false
        }

        chomped = _bits[row - 1][column - 1];  # 将chomped赋值为_bits数组中对应行列的元素值

        for (int r = row; r <= _rowCount; r++)  # 循环，从row到_rowCount
        {
            for (int c = column; c <= _columnCount; c++)  # 循环，从column到_columnCount
            {
                _bits[r - 1][c - 1] = ' ';  # 将_bits数组中对应行列的元素值赋值为' '
            }
        }
```
这是一个类的结束标记。

```csharp
        return true;
```
返回一个布尔值true。

```csharp
    public override string ToString()
    {
        var builder = new StringBuilder().AppendLine("       1 2 3 4 5 6 7 8 9");
```
创建一个StringBuilder对象，并向其添加一行文本。

```csharp
        for (int row = 1; row <= _bits.Length; row++)
        {
            builder.Append(' ').Append(row).Append("     ").AppendLine(string.Join(' ', _bits[row - 1]));
        }
```
使用循环遍历_bits数组中的元素，并将其添加到StringBuilder对象中。

```csharp
        return builder.ToString();
```
将StringBuilder对象转换为字符串并返回。
```