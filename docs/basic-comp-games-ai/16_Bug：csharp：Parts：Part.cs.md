# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\Part.cs`

```
using BugGame.Resources;  # 导入 BugGame.Resources 命名空间

namespace BugGame.Parts;  # 声明 BugGame.Parts 命名空间

internal class Part : IPart  # 声明 Part 类，并实现 IPart 接口
{
    private readonly Message _addedMessage;  # 声明私有只读字段 _addedMessage，类型为 Message
    private readonly Message _duplicateMessage;  # 声明私有只读字段 _duplicateMessage，类型为 Message

    public Part(Message addedMessage, Message duplicateMessage)  # 声明 Part 类的构造函数，接受两个 Message 类型的参数
    {
        _addedMessage = addedMessage;  # 将构造函数参数 addedMessage 赋值给 _addedMessage 字段
        _duplicateMessage = duplicateMessage;  # 将构造函数参数 duplicateMessage 赋值给 _duplicateMessage 字段
    }

    public virtual bool IsComplete => IsPresent;  # 声明虚属性 IsComplete，返回 IsPresent 的值

    protected bool IsPresent { get; private set; }  # 声明受保护的属性 IsPresent，可读写

    public string Name => GetType().Name;  # 实现 Name 属性，返回当前对象的类型名
    # 尝试向对象中添加消息，如果对象已经存在消息，则返回失败并返回重复消息
    def TryAdd(self, message):
        # 如果对象已经存在消息
        if self.IsPresent:
            # 将重复消息赋值给传入的消息变量
            message = self._duplicateMessage
            # 返回失败
            return False
        # 将添加消息赋值给传入的消息变量
        message = self._addedMessage
        # 将对象的存在状态设置为True
        self.IsPresent = True
        # 返回成功
        return True
```