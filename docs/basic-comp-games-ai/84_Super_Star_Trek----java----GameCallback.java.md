# `basic-computer-games\84_Super_Star_Trek\java\GameCallback.java`

```
/**
 * 游戏回调接口，用于将控制反转从GalaxyMap和Enterprise解耦到游戏类中。
 */
public interface GameCallback {
    // 进入新的象限
    void enterNewQuadrant();
    // 增加星日期
    void incrementStardate(double increment);
    // 游戏成功结束
    void endGameSuccess();
    // 游戏失败结束，参数表示Enterprise是否被摧毁
    void endGameFail(boolean enterpriseDestroyed);
}
*/
```