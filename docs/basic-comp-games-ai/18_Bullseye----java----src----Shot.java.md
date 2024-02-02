# `basic-computer-games\18_Bullseye\java\src\Shot.java`

```py
/**
 * This class records the percentage chance of a given type of shot
 * scoring specific points
 * see Bullseye class points calculation method where its used
 */
public class Shot {

    double[] chances; // 用于记录特定类型的射击得分的百分比机会的数组

    // 为特定类型的射击传递一个双精度数组
    Shot(double[] shots) {
        chances = new double[shots.length]; // 初始化机会数组
        System.arraycopy(shots, 0, chances, 0, shots.length); // 复制传入的数组到机会数组
    }

    // 获取特定索引位置的射击机会
    public double getShot(int index) {
        return chances[index];
    }
}
```