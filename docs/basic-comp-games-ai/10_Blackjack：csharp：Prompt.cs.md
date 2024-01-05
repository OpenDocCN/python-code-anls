# `10_Blackjack\csharp\Prompt.cs`

```
using System;  # 导入 System 模块

namespace Blackjack:  # 定义名为 Blackjack 的命名空间
    public static class Prompt:  # 定义名为 Prompt 的公共静态类
        public static bool ForYesNo(string prompt):  # 定义名为 ForYesNo 的公共静态方法，接受一个字符串参数 prompt
            while(true):  # 进入无限循环
                Console.Write("{0} ", prompt);  # 在控制台打印提示信息
                var input = Console.ReadLine();  # 从控制台读取用户输入并赋值给变量 input
                if (input.StartsWith("y", StringComparison.InvariantCultureIgnoreCase)):  # 如果用户输入以 "y" 开头（不区分大小写）
                    return true;  # 返回 true
                if (input.StartsWith("n", StringComparison.InvariantCultureIgnoreCase)):  # 如果用户输入以 "n" 开头（不区分大小写）
                    return false;  # 返回 false
                WriteNotUnderstood();  # 调用 WriteNotUnderstood 方法
                // 提示用户输入一个字符
                // 如果输入的字符不在允许的字符列表中，则继续提示用户输入
                if (input.Length != 1 || !allowedCharacters.Contains(input[0]))
                    WriteNotUnderstood();
                else
                    return input;
            }
        }
                if (input.Length > 0)
                {
                    // 检查输入是否有内容
                    var character = input.Substring(0, 1);
                    // 获取输入的第一个字符
                    var characterIndex = allowedCharacters.IndexOf(character, StringComparison.InvariantCultureIgnoreCase);
                    // 在允许的字符列表中查找输入字符的索引
                    if (characterIndex != -1)
                        // 如果输入字符在允许的字符列表中存在
                        return allowedCharacters.Substring(characterIndex, 1);
                        // 返回允许的字符列表中与输入字符对应的字符
                }

                // 如果输入为空或者输入字符不在允许的字符列表中
                Console.WriteLine("Type one of {0} please", String.Join(", ", allowedCharacters.ToCharArray()));
                // 提示用户输入允许的字符列表中的一个字符
            }
        }

        private static void WriteNotUnderstood()
        {
            // 输出提示信息，表示无法理解用户的输入
            Console.WriteLine("Sorry, I didn't understand.");
        }
    }
}
```