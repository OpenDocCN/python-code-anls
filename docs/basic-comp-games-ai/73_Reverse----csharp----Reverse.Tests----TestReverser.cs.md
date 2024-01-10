# `basic-computer-games\73_Reverse\csharp\Reverse.Tests\TestReverser.cs`

```
// 声明命名空间为 Reverse.Tests
namespace Reverse.Tests
{
    // 声明 TestReverser 类，继承自 Reverser 类
    internal class TestReverser : Reverser
    {
        // 声明 TestReverser 类的构造函数，接受一个整数参数 arraySize，并调用基类 Reverser 的构造函数
        public TestReverser(int arraySize) : base(arraySize) { }

        // 声明 GetArray 方法，用于获取私有成员变量 _array
        public int[] GetArray()
        {
            return _array;
        }

        // 声明 SetArray 方法，用于设置私有成员变量 _array
        public void SetArray(int[] array)
        {
            _array = array;
        }
    }
}
```