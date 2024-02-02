# `basic-computer-games\90_Tower\csharp\UI\Prompt.cs`

```py
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间，用于使用泛型集合
using System.Linq;  // 导入 System.Linq 命名空间，用于使用 LINQ 查询
using static Tower.Resources.Strings;  // 导入 Tower.Resources.Strings 类中的静态成员

namespace Tower.UI  // 声明 Tower.UI 命名空间
{
    internal class Prompt  // 声明内部类 Prompt
    {
        public static Prompt DiskCount =  // 声明静态的 Prompt 对象 DiskCount
            new(DiskCountPrompt, DiskCountRetry, DiskCountQuit, 1, 2, 3, 4, 5, 6, 7) { RetriesAllowed = 2 };  // 使用 DiskCountPrompt, DiskCountRetry, DiskCountQuit 作为参数创建 Prompt 对象，并设置 RetriesAllowed 属性为 2

        public static Prompt Disk =  // 声明静态的 Prompt 对象 Disk
            new(DiskPrompt, DiskRetry, DiskQuit, 3, 5, 7, 9, 11, 13, 15) { RepeatPrompt = false };  // 使用 DiskPrompt, DiskRetry, DiskQuit 作为参数创建 Prompt 对象，并设置 RepeatPrompt 属性为 false

        public static Prompt Needle = new(NeedlePrompt, NeedleRetry, NeedleQuit, 1, 2, 3);  // 声明静态的 Prompt 对象 Needle，使用 NeedlePrompt, NeedleRetry, NeedleQuit 作为参数创建 Prompt 对象

        private readonly HashSet<int> _validValues;  // 声明私有的只读整型哈希集合 _validValues

        private Prompt(string prompt, string retryMessage, string quitMessage, params int[] validValues)  // 声明私有的构造函数 Prompt，接受 prompt、retryMessage、quitMessage 和可变长度的整型数组 validValues 作为参数
        {
            Message = prompt;  // 设置 Message 属性为 prompt
            RetryMessage = retryMessage;  // 设置 RetryMessage 属性为 retryMessage
            QuitMessage = quitMessage;  // 设置 QuitMessage 属性为 quitMessage
            _validValues = validValues.ToHashSet();  // 使用 validValues 创建整型哈希集合并赋值给 _validValues
            RetriesAllowed = 1;  // 设置 RetriesAllowed 属性为 1
            RepeatPrompt = true;  // 设置 RepeatPrompt 属性为 true
        }

        public string Message { get; }  // 声明只读的字符串属性 Message
        public string RetryMessage { get; }  // 声明只读的字符串属性 RetryMessage
        public string QuitMessage { get; }  // 声明只读的字符串属性 QuitMessage
        public int RetriesAllowed { get; private set; }  // 声明可读写的整型属性 RetriesAllowed
        public bool RepeatPrompt { get; private set; }  // 声明可读写的布尔属性 RepeatPrompt

        public bool TryValidateResponse(float number, out int integer)  // 声明公共的方法 TryValidateResponse，接受浮点数 number 和整型数 integer 的引用作为参数
        {
            integer = (int)number;  // 将 number 转换为整型并赋值给 integer
            return integer == number && _validValues.Contains(integer);  // 返回整型数是否等于浮点数并且在 _validValues 中
        }
    }
}
```