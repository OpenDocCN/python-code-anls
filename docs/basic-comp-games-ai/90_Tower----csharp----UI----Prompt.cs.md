# `basic-computer-games\90_Tower\csharp\UI\Prompt.cs`

```

// 引入必要的命名空间
using System.Collections.Generic;
using System.Linq;
using static Tower.Resources.Strings;

// 声明 Tower.UI 命名空间下的 Prompt 类
namespace Tower.UI
{
    // 声明 Prompt 类
    internal class Prompt
    {
        // 声明并初始化 DiskCount 静态属性
        public static Prompt DiskCount =
            new(DiskCountPrompt, DiskCountRetry, DiskCountQuit, 1, 2, 3, 4, 5, 6, 7) { RetriesAllowed = 2 };

        // 声明并初始化 Disk 静态属性
        public static Prompt Disk =
            new(DiskPrompt, DiskRetry, DiskQuit, 3, 5, 7, 9, 11, 13, 15) { RepeatPrompt = false };

        // 声明并初始化 Needle 静态属性
        public static Prompt Needle = new(NeedlePrompt, NeedleRetry, NeedleQuit, 1, 2, 3);

        // 声明私有字段 _validValues
        private readonly HashSet<int> _validValues;

        // 声明 Prompt 类的构造函数
        private Prompt(string prompt, string retryMessage, string quitMessage, params int[] validValues)
        {
            // 初始化 Message、RetryMessage、QuitMessage 和 _validValues 字段
            Message = prompt;
            RetryMessage = retryMessage;
            QuitMessage = quitMessage;
            _validValues = validValues.ToHashSet();
            RetriesAllowed = 1;
            RepeatPrompt = true;
        }

        // 声明 Message 属性
        public string Message { get; }
        // 声明 RetryMessage 属性
        public string RetryMessage { get; }
        // 声明 QuitMessage 属性
        public string QuitMessage { get; }
        // 声明 RetriesAllowed 属性
        public int RetriesAllowed { get; private set; }
        // 声明 RepeatPrompt 属性
        public bool RepeatPrompt { get; private set; }

        // 声明 TryValidateResponse 方法
        public bool TryValidateResponse(float number, out int integer)
        {
            // 尝试将浮点数转换为整数
            integer = (int)number;
            // 判断转换后的整数是否等于原始浮点数，并且是否在_validValues中
            return integer == number && _validValues.Contains(integer);
        }
    }
}

```