# `.\PaddleOCR\deploy\android_demo\app\src\androidTest\java\com\baidu\paddle\lite\demo\ocr\ExampleInstrumentedTest.java`

```py
/**
 * 导入必要的类库
 */
package com.baidu.paddle.lite.demo.ocr;

/**
 * 导入 Android 相关类库
 */
import android.content.Context;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

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
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    @Test
    public void useAppContext() {
        // 获取被测试应用的上下文
        Context appContext = InstrumentationRegistry.getTargetContext();

        // 断言被测试应用的包名是否为指定值
        assertEquals("com.baidu.paddle.lite.demo", appContext.getPackageName());
    }
}
```