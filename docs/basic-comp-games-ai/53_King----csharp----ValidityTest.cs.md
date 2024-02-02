# `basic-computer-games\53_King\csharp\ValidityTest.cs`

```py
// 命名空间 King 下的 ValidityTest 类
namespace King
{
    // ValidityTest 类
    internal class ValidityTest
    {
        // 只读字段，用于存储 float 类型的断言函数
        private readonly Predicate<float> _isValid;
        // 只读字段，用于存储返回错误信息的函数
        private readonly Func<string> _getError;

        // 构造函数，接受一个断言函数和一个错误信息字符串
        public ValidityTest(Predicate<float> isValid, string error)
            : this(isValid, () => error)
        {
        }

        // 构造函数，接受一个断言函数和一个返回错误信息的函数
        public ValidityTest(Predicate<float> isValid, Func<string> getError)
        {
            // 将传入的断言函数赋值给 _isValid 字段
            _isValid = isValid;
            // 将传入的返回错误信息的函数赋值给 _getError 字段
            _getError = getError;
        }

        // 判断给定的值是否有效
        public bool IsValid(float value, IReadWrite io)
        {
            // 如果断言函数返回 true，则返回 true
            if (_isValid(value)) { return true; }
            
            // 否则，将错误信息写入 IReadWrite 对象，并返回 false
            io.Write(_getError());
            return false;
        }
    }
}
```