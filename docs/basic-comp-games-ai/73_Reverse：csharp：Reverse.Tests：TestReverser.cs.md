# `73_Reverse\csharp\Reverse.Tests\TestReverser.cs`

```
# 定义一个内部类 TestReverser，继承自 Reverser 类
namespace Reverse.Tests
{
    internal class TestReverser : Reverser
    {
        # 定义 TestReverser 类的构造函数，接受一个整数参数 arraySize，并调用父类 Reverser 的构造函数
        public TestReverser(int arraySize) : base(arraySize) { }

        # 定义一个公共方法 GetArray，用于返回私有成员变量 _array
        public int[] GetArray()
        {
            return _array;
        }

        # 定义一个公共方法 SetArray，用于设置私有成员变量 _array 的值
        public void SetArray(int[] array)
        {
            _array = array;
        }
    }
}
```