# `90_Tower\csharp\UI\Prompt.cs`

```
using System.Collections.Generic; // 导入 System.Collections.Generic 命名空间，用于使用 HashSet 类型
using System.Linq; // 导入 System.Linq 命名空间，用于使用 LINQ 查询
using static Tower.Resources.Strings; // 导入 Tower.Resources.Strings 命名空间中的静态成员

namespace Tower.UI
{
    internal class Prompt
    {
        // 创建一个名为 DiskCount 的静态 Prompt 对象，使用指定的参数进行初始化
        public static Prompt DiskCount =
            new(DiskCountPrompt, DiskCountRetry, DiskCountQuit, 1, 2, 3, 4, 5, 6, 7) { RetriesAllowed = 2 };

        // 创建一个名为 Disk 的静态 Prompt 对象，使用指定的参数进行初始化
        public static Prompt Disk =
            new(DiskPrompt, DiskRetry, DiskQuit, 3, 5, 7, 9, 11, 13, 15) { RepeatPrompt = false };

        // 创建一个名为 Needle 的静态 Prompt 对象，使用指定的参数进行初始化
        public static Prompt Needle = new(NeedlePrompt, NeedleRetry, NeedleQuit, 1, 2, 3);

        private readonly HashSet<int> _validValues; // 声明一个名为 _validValues 的只读 HashSet<int> 字段

        // 构造函数，接受 prompt、retryMessage、quitMessage 和 validValues 参数
        private Prompt(string prompt, string retryMessage, string quitMessage, params int[] validValues)
        {
            Message = prompt;  # 将参数 prompt 赋值给属性 Message
            RetryMessage = retryMessage;  # 将参数 retryMessage 赋值给属性 RetryMessage
            QuitMessage = quitMessage;  # 将参数 quitMessage 赋值给属性 QuitMessage
            _validValues = validValues.ToHashSet();  # 将参数 validValues 转换为 HashSet 类型并赋值给属性 _validValues
            RetriesAllowed = 1;  # 将属性 RetriesAllowed 设置为 1
            RepeatPrompt = true;  # 将属性 RepeatPrompt 设置为 true
        }

        public string Message { get; }  # 定义只读属性 Message
        public string RetryMessage { get; }  # 定义只读属性 RetryMessage
        public string QuitMessage { get; }  # 定义只读属性 QuitMessage
        public int RetriesAllowed { get; private set; }  # 定义属性 RetriesAllowed，并设置为私有可写
        public bool RepeatPrompt { get; private set; }  # 定义属性 RepeatPrompt，并设置为私有可写

        public bool TryValidateResponse(float number, out int integer)  # 定义方法 TryValidateResponse，接受一个浮点数参数和一个整数参数的引用
        {
            integer = (int)number;  # 将浮点数转换为整数并赋值给参数 integer
            return integer == number && _validValues.Contains(integer);  # 返回判断结果，判断整数是否等于原始浮点数并且 _validValues 中是否包含该整数
        }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```