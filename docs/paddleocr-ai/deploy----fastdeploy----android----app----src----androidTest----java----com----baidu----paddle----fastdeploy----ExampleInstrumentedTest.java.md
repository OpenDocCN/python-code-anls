# `.\PaddleOCR\deploy\fastdeploy\android\app\src\androidTest\java\com\baidu\paddle\fastdeploy\ExampleInstrumentedTest.java`

```
/**
 * 导入必要的类库
 */
package com.baidu.paddle.fastdeploy;

/**
 * 导入 Android 相关类库
 */
import android.content.Context;

/**
 * 导入测试相关类库
 */
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

/**
 * 导入测试相关类库
 */
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * 导入断言相关类库
 */
import static org.junit.Assert.*;

/**
 * 测试类，运行在 Android 设备上
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    /**
     * 测试方法，验证应用上下文
     */
    @Test
    public void useAppContext() {
        // 获取被测试应用的上下文
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        // 断言应用包名是否为指定值
        assertEquals("com.baidu.paddle.fastdeploy", appContext.getPackageName());
    }
}
```