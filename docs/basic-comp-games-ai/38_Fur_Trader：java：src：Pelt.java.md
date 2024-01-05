# `d:/src/tocomm/basic-computer-games\38_Fur_Trader\java\src\Pelt.java`

```
# 定义一个名为 Pelt 的类，用于跟踪玩家拥有的特定皮毛类型的名称和数量
class Pelt:

    # 初始化方法，设置皮毛的名称和数量
    def __init__(self, name, number):
        self.name = name
        self.number = number

    # 设置皮毛的数量
    def setPeltCount(self, pelts):
        self.number = pelts

    # 获取皮毛的数量
    def getNumber(self):
        return self.number
# 返回对象的名称
public String getName() {
    return this.name;
}

# 重置对象的number属性为0
public void lostPelts() {
    this.number = 0;
}
```