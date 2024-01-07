# `basic-computer-games\16_Bug\csharp\Parts\PartCollection.cs`

```

// 使用 System.Text 命名空间
using System.Text;
// 使用 BugGame.Resources 命名空间
using BugGame.Resources;

// 创建 PartCollection 类
namespace BugGame.Parts
{
    // 创建 PartCollection 类
    internal class PartCollection
    {
        // 声明私有变量 _maxCount，_addedMessage，_fullMessage，_count
        private readonly int _maxCount;
        private readonly Message _addedMessage;
        private readonly Message _fullMessage;
        private int _count;

        // 创建 PartCollection 类的构造函数
        public PartCollection(int maxCount, Message addedMessage, Message fullMessage)
        {
            // 初始化私有变量
            _maxCount = maxCount;
            _addedMessage = addedMessage;
            _fullMessage = fullMessage;
        }

        // 创建 IsComplete 属性，判断是否已满
        public bool IsComplete => _count == _maxCount;

        // 创建 TryAddOne 方法，尝试添加一个部件
        public bool TryAddOne(out Message message)
        {
            // 如果还有空位
            if (_count < _maxCount)
            {
                // 增加部件数量
                _count++;
                // 返回添加部件的消息
                message = _addedMessage.ForValue(_count);
                return true;
            }

            // 返回已满的消息
            message = _fullMessage;
            return false;
        }

        // 创建 AppendTo 方法，向 StringBuilder 中添加内容
        protected void AppendTo(StringBuilder builder, int offset, int length, char character)
        {
            // 如果部件数量为 0，则直接返回
            if (_count == 0) { return; }

            // 循环向 StringBuilder 中添加内容
            for (var i = 0; i < length; i++)
            {
                builder.Append(' ', offset);

                for (var j = 0; j < _count; j++)
                {
                    builder.Append(character).Append(' ');
                }
                builder.AppendLine();
            }
        }
    }
}

```