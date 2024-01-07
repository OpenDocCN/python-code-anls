# `basic-computer-games\18_Bullseye\java\src\Shot.java`

```

/**
 * 这个类记录了特定类型的投篮得分特定点数的百分比机会
 * 参见Bullseye类中使用的得分计算方法
 */
public class Shot {

    double[] chances; // 存储特定类型的投篮得分百分比的数组

    // 为特定类型的投篮传递一个双精度数组
    Shot(double[] shots) {
        chances = new double[shots.length]; // 初始化chances数组
        System.arraycopy(shots, 0, chances, 0, shots.length); // 复制传入的数组到chances数组
    }

    // 获取特定索引位置的投篮得分百分比
    public double getShot(int index) {
        return chances[index];
    }
}

```