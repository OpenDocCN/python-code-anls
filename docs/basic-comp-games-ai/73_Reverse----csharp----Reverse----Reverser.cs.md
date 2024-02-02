# `basic-computer-games\73_Reverse\csharp\Reverse\Reverser.cs`

```py
// 命名空间 Reverse，包含 Reverser 类
namespace Reverse
{
    // Reverser 类，用于数组反转操作
    public class Reverser
    {
        // 保护类型的整型数组 _array
        protected int[] _array;

        // 构造函数，根据数组大小创建随机数组
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

            // 遍历索引位置之前的元素，进行反转操作
            for (int i = 0; i < index / 2; i++)
            {
                int temp = _array[i];
                int upperIndex = index - 1 - i;
                _array[i] = _array[upperIndex];
                _array[upperIndex] = temp;
            }
        }

        // 判断数组是否按升序排列
        public bool IsArrayInAscendingOrder()
        {
            // 遍历数组，如果出现逆序则返回 false
            for (int i = 1; i < _array.Length; i++)
            {
                if (_array[i] < _array[i - 1])
                {
                    return false;
                }
            }

            // 数组按升序排列，返回 true
            return true;
        }

        // 创建指定大小的随机数组
        private int[] CreateRandomArray(int size)
        {
            // 如果数组大小小于 1，则抛出异常
            if (size < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "Array size must be a positive integer");
            }

            // 创建大小为 size 的整型数组
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
            // 使用 StringBuilder 构建数组的字符串表示
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