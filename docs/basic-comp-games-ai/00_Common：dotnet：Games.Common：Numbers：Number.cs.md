# `d:/src/tocomm/basic-computer-games\00_Common\dotnet\Games.Common\Numbers\Number.cs`

```
namespace Games.Common.Numbers;

/// <summary>
/// A single-precision floating-point number with string formatting equivalent to the BASIC interpreter.
/// </summary>
public struct Number
{
    private readonly float _value; // 声明一个私有的单精度浮点数变量 _value

    public Number (float value) // 构造函数，用于初始化 _value
    {
        _value = value;
    }

    public static implicit operator float(Number value) => value._value; // 定义从 Number 到 float 的隐式转换

    public static implicit operator Number(float value) => new Number(value); // 定义从 float 到 Number 的隐式转换

    public override string ToString() => _value < 0 ? $"{_value} " : $" {_value} "; // 重写 ToString 方法，根据 _value 的值返回对应的字符串
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```