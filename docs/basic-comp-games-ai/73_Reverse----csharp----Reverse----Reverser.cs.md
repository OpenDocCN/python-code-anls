# `basic-computer-games\73_Reverse\csharp\Reverse\Reverser.cs`

```

// 引入命名空间
using System;
using System.Text;

// 定义一个名为Reverser的公共类
namespace Reverse
{
    public class Reverser
    {
        // 声明一个受保护的整型数组
        protected int[] _array;

        // 构造函数，接受一个整数参数作为数组大小，创建一个随机数组并赋值给_array
        public Reverser(int arraySize)
        {
            _array = CreateRandomArray(arraySize);
        }

        // 反转数组中指定索引位置之前的元素
        public void Reverse(int index)
        {
            // 如果索引超出数组长度，则直接返回
            if (index > _array.Length)
            {
                return;
            }

            // 循环反转数组元素
            for (int i = 0; i < index / 2; i++)
            {
                int temp = _array[i];
                int upperIndex = index - 1 - i;
                _array[i] = _array[upperIndex];
                _array[upperIndex] = temp;
            }
        }

        // 检查数组是否按升序排列
        public bool IsArrayInAscendingOrder()
        {
            // 遍历数组，如果发现有元素比前一个元素小，则返回false
            for (int i = 1; i < _array.Length; i++)
            {
                if (_array[i] < _array[i - 1])
                {
                    return false;
                }
            }

            // 如果遍历完数组都没有发现逆序对，则返回true
            return true;
        }

        // 创建一个指定大小的随机数组
        private int[] CreateRandomArray(int size)
        {
            // 如果数组大小小于1，则抛出参数异常
            if (size < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "Array size must be a positive integer");
            }

            // 创建一个大小为size的数组，并按顺序填充
            var array = new int[size];
            for (int i = 1; i <= size; i++)
            {
                array[i - 1] = i;
            }

            // 使用随机数打乱数组顺序
            var rnd = new Random();
            for (int i = size; i > 1;)
            {
                int k = rnd.Next(i);
                --i;
                int temp = array[i];
                array[i] = array[k];
                array[k] = temp;
            }
            return array;
        }

        // 返回数组的字符串表示形式
        public string GetArrayString()
        {
            // 使用StringBuilder构建数组的字符串表示
            var sb = new StringBuilder();
            foreach (int i in _array)
            {
                sb.Append(" " + i + " ");
            }
            return sb.ToString();
        }
    }
}

```