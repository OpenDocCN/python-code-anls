# `05_Bagels\csharp\BagelNumber.cs`

```
# 定义枚举类型 BagelValidation，包含 Valid、WrongLength、NotUnique、NonDigit 四种取值
# 定义类 BagelNumber
# 创建静态只读的 Random 对象 Rnd
# 创建只读的整型数组 _digits
# 重写 ToString 方法
		return String.Join('-', _digits);
```
这行代码的作用是将_digits列表中的元素以"-"连接起来，然后返回连接后的字符串。

```
		public static BagelNumber CreateSecretNumber(int numDigits)
	{
		if (numDigits < 3 || numDigits > 9)
			throw new ArgumentOutOfRangeException(nameof(numDigits),
				"Number of digits must be between 3 and 9, inclusive");

		var digits = GetDigits(numDigits);
		return new BagelNumber(digits);
	}
```
这段代码定义了一个静态方法CreateSecretNumber，用于创建一个包含指定位数的秘密数字。如果指定的位数不在3到9之间，会抛出ArgumentOutOfRangeException异常。然后调用GetDigits方法获取指定位数的数字，并用这些数字创建一个BagelNumber对象。

```
		public static BagelValidation IsValid(string number, int length)
	{
		if (number.Length != length)
			return BagelValidation.WrongLength;
```
这段代码定义了一个静态方法IsValid，用于验证输入的数字是否符合指定的长度。如果输入的数字长度不等于指定的长度，就返回BagelValidation.WrongLength。

			// 如果输入的数字中包含非数字字符，则返回非数字的验证结果
			if (!number.All(Char.IsDigit))
				return BagelValidation.NonDigit;

			// 如果输入的数字中包含重复的数字，则返回不唯一的验证结果
			if (new HashSet<char>(number).Count != length)
				return BagelValidation.NotUnique;

			// 如果输入的数字符合要求，则返回有效的验证结果
			return BagelValidation.Valid;
		}

		// 构造函数，用于初始化 BagelNumber 对象
		public BagelNumber(string number)
		{
			// 如果输入的数字中包含非数字字符，则抛出参数异常
			if (number.Any(d => !Char.IsDigit(d)))
				throw new ArgumentException("Number must be all unique digits", nameof(number));

			// 将输入的数字转换为整数数组并赋值给 _digits
			_digits = number.Select(d => d - '0').ToArray();
		}

		//public BagelNumber(long number)
		//{
		//	var digits = new List<int>();
		// 如果数字大于等于 1E10（即10的10次方），抛出参数超出范围的异常
		//	throw new ArgumentOutOfRangeException(nameof(number), "Number can be no more than 9 digits");

		// 当数字大于 0 时，执行循环
		//	{
		//		long num = number / 10; // 将数字除以 10，得到商
		//		int digit = (int)(number - (num * 10)); // 计算出个位数
		//		number = num; // 更新数字为商
		//		digits.Add(digit); // 将个位数添加到列表中
		//	}

		//	_digits = digits.ToArray(); // 将列表转换为数组，并赋值给 _digits
		//}

		// 通过给定的数字数组构造 BagelNumber 对象
		public BagelNumber(int[] digits)
		{
			_digits = digits; // 将给定的数字数组赋值给 _digits
		}

		// 获取指定位数的数字数组
		private static  int[] GetDigits(int numDigits)
		{
			# 创建一个整数数组
			int[] digits = {1, 2, 3, 4, 5, 6, 7, 8, 9};
			# 调用Shuffle函数对数组进行随机排列
			Shuffle(digits);
			# 返回数组中指定数量的元素
			return digits.Take(numDigits).ToArray();

		}

		# 定义Shuffle函数，用于对数组进行随机排列
		private static void Shuffle(int[] digits)
		{
			# 从数组末尾向前遍历
			for (int i = digits.Length - 1; i > 0; --i)
			{
				# 生成一个小于i的随机数
				int pos = Rnd.Next(i);
				# 交换数组中的元素位置
				var t = digits[i];
				digits[i] = digits[pos];
				digits[pos] = t;
			}

		}

		# 定义CompareTo函数，用于比较两个BagelNumber对象
		public (int pico, int fermi) CompareTo(BagelNumber other)
# 初始化 pico 和 fermi 变量，用于记录猜测数字和目标数字的匹配情况
int pico = 0;
int fermi = 0;

# 循环遍历猜测数字和目标数字的每一位
for (int i = 0; i < _digits.Length; i++)
{
    for (int j = 0; j < other._digits.Length; j++)
    {
        # 如果猜测数字的某一位和目标数字的某一位相等
        if (_digits[i] == other._digits[j])
        {
            # 如果位置也相等，则增加 fermi 计数
            if (i == j)
                ++fermi;
            # 如果位置不相等，则增加 pico 计数
            else
                ++pico;
        }
    }
}

# 返回 pico 和 fermi 的计数结果
return (pico, fermi);
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```