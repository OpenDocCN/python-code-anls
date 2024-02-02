# `basic-computer-games\10_Blackjack\java\test\UserIoTest.java`

```py
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

public class UserIoTest {

    @ParameterizedTest(name = "''{0}'' is accepted as ''no''")
    @ValueSource(strings = {"N", "n", "No", "NO", "no"})
    public void testPromptBooleanAcceptsNo(String response) {
        // Given
        // 创建一个字符串读取器，用于模拟用户输入
        Reader in = new StringReader(response + "\n");
        // 创建一个字符串写入器，用于捕获输出
        StringWriter out = new StringWriter();
        // 创建用户输入输出对象
        UserIo userIo = new UserIo(in, out);

        // When
        // 调用用户输入输出对象的方法，获取结果
        boolean result = userIo.promptBoolean("TEST");

        // Then
        // 检查输出是否符合预期
        assertEquals("TEST? ", out.toString());
        // 检查结果是否符合预期
        assertFalse(result);
    }

    @ParameterizedTest(name = "''{0}'' is accepted as ''yes''")
    @ValueSource(strings = {"Y", "y", "Yes", "YES", "yes", "", "foobar"})
    public void testPromptBooleanAcceptsYes(String response) {
        // Given
        // 创建一个字符串读取器，用于模拟用户输入
        Reader in = new StringReader(response + "\n");
        // 创建一个字符串写入器，用于捕获输出
        StringWriter out = new StringWriter();
        // 创建用户输入输出对象
        UserIo userIo = new UserIo(in, out);

        // When
        // 调用用户输入输出对象的方法，获取结果
        boolean result = userIo.promptBoolean("TEST");

        // Then
        // 检查输出是否符合预期
        assertEquals("TEST? ", out.toString());
        // 检查结果是否符合预期
        assertTrue(result);
    }

    @ParameterizedTest(name = "''{0}'' is accepted as number")
    @CsvSource({
        "1,1",
        "0,0",
        "-1,-1",
    })
    // 定义测试方法，验证 promptInt 方法能够接受数字输入
    public void testPromptIntAcceptsNumbers(String response, int expected) {
        // 准备测试数据，将字符串转换为字符流和字符串输出流
        Reader in = new StringReader(response + "\n");
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);

        // 调用被测试的方法
        int result = userIo.promptInt("TEST");

        // 验证方法是否按预期输出了提示信息和返回了正确的结果
        assertEquals("TEST? ", out.toString());
        assertEquals(expected, result);
    }

    // 定义测试方法，验证 promptInt 方法在接收到非数字输入时能够打印错误信息并重新提示
    @Test
    @DisplayName("promptInt should print an error and reprompt if given a non-numeric response")
    public void testPromptIntRepromptsOnNonNumeric() {
        // 准备测试数据，将字符串转换为字符流和字符串输出流
        Reader in = new StringReader("foo" + System.lineSeparator() +"1"); // word, then number
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);

        // 调用被测试的方法
        int result = userIo.promptInt("TEST");

        // 验证方法是否按预期输出了错误信息并重新提示，并返回了正确的结果
        assertEquals("TEST? !NUMBER EXPECTED - RETRY INPUT LINE" + System.lineSeparator() +"? ", out.toString());
        assertEquals(1, result);
    }
# 闭合前面的函数定义
```