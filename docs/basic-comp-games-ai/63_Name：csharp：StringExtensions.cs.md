# `63_Name\csharp\StringExtensions.cs`

```
        // 设置默认控制台宽度为120
        private const int ConsoleWidth = 120; // default console width

        // 创建字符串居中对齐的扩展方法
        public static string CentreAlign(this string value)
        {
            // 计算需要添加的空格数
            int spaces = ConsoleWidth - value.Length;
            // 计算左边填充的空格数
            int leftPadding = spaces / 2 + value.Length;

            // 返回居中对齐的字符串
            return value.PadLeft(leftPadding).PadRight(ConsoleWidth);
        }

        // 创建字符串反转的扩展方法
        public static string Reverse(this string value)
        {
            // 检查字符串是否为空
            if (value is null)
            {
        public static string Reverse(this string value)
        {
            // 如果输入的字符串为空，则返回空
            if (value is null)
            {
                return null;
            }
            // 将字符串转换为字符数组，并反转数组中的字符顺序
            char[] characterArray = value.ToCharArray();
            Array.Reverse(characterArray);
            // 将反转后的字符数组转换为字符串并返回
            return new String(characterArray);
        }

        public static string Sort(this string value)
        {
            // 如果输入的字符串为空，则返回空
            if (value is null)
            {
                return null;
            }
            // 将字符串转换为字符数组，并对数组中的字符进行排序
            char[] characters = value.ToCharArray();
            Array.Sort(characters);
            // 将排序后的字符数组转换为字符串并返回
            return new string(characters);
        }
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```