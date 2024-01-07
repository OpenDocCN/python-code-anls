# `basic-computer-games\73_Reverse\csharp\Reverse.Tests\TestReverser.cs`

```

// 命名空间 Reverse.Tests，用于组织和管理类
namespace Reverse.Tests
{
    // TestReverser 类继承自 Reverser 类
    internal class TestReverser : Reverser
    {
        // 构造函数，接受一个整数参数作为数组大小，并调用基类的构造函数进行初始化
        public TestReverser(int arraySize) : base(arraySize) { }

        // 获取数组的方法，返回私有成员变量 _array
        public int[] GetArray()
        {
            return _array;
        }

        // 设置数组的方法，接受一个整型数组作为参数，并将私有成员变量 _array 设置为该数组
        public void SetArray(int[] array)
        {
            _array = array;
        }
    }
}

```