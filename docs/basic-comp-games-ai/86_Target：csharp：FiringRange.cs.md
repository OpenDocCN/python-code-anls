# `86_Target\csharp\FiringRange.cs`

```
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 命名空间，以便使用其中的类和方法

namespace Target  # 命名空间 Target
{
    internal class FiringRange  # 内部类 FiringRange
    {
        private readonly IRandom _random;  # 声明一个只读的 IRandom 接口类型的变量 _random
        private Point _targetPosition;  # 声明一个 Point 类型的变量 _targetPosition

        public FiringRange(IRandom random)  # FiringRange 类的构造函数，接受一个 IRandom 类型的参数 random
        {
            _random = random;  # 将参数 random 赋值给 _random 变量
        }

        public Point NextTarget() =>  _targetPosition = _random.NextPosition();  # 定义一个方法 NextTarget，返回值为 Point 类型，将 _targetPosition 赋值为 _random.NextPosition() 的结果

        public Explosion Fire(Angle angleFromX, Angle angleFromZ, float distance)  # 定义一个方法 Fire，接受 Angle 类型的 angleFromX 和 angleFromZ 参数，以及 float 类型的 distance 参数
        {
            var explosionPosition = new Point(angleFromX, angleFromZ, distance);  # 声明一个 Point 类型的变量 explosionPosition，并赋值为使用 angleFromX、angleFromZ 和 distance 创建的新 Point 对象
            var targetOffset = explosionPosition - _targetPosition;  # 声明一个变量 targetOffset，赋值为 explosionPosition 减去 _targetPosition 的结果
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```