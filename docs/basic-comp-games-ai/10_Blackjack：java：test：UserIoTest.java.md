# `d:/src/tocomm/basic-computer-games\10_Blackjack\java\test\UserIoTest.java`

```
import static org.junit.jupiter.api.Assertions.assertEquals;  # 导入断言方法，用于断言测试结果是否符合预期
import static org.junit.jupiter.api.Assertions.assertFalse;  # 导入断言方法，用于断言测试结果是否为假
import static org.junit.jupiter.api.Assertions.assertTrue;   # 导入断言方法，用于断言测试结果是否为真

import java.io.Reader;  # 导入 Reader 类，用于读取字符流
import java.io.StringReader;  # 导入 StringReader 类，用于读取字符串
import java.io.StringWriter;  # 导入 StringWriter 类，用于写入字符串

import org.junit.jupiter.api.DisplayName;  # 导入 DisplayName 注解，用于指定测试用例的显示名称
import org.junit.jupiter.api.Test;  # 导入 Test 注解，用于标记测试方法
import org.junit.jupiter.params.ParameterizedTest;  # 导入 ParameterizedTest 注解，用于标记参数化测试方法
import org.junit.jupiter.params.provider.CsvSource;  # 导入 CsvSource 注解，用于提供参数化测试的参数
import org.junit.jupiter.params.provider.ValueSource;  # 导入 ValueSource 注解，用于提供参数化测试的参数

public class UserIoTest {

    @ParameterizedTest(name = "''{0}'' is accepted as ''no''")  # 使用 ParameterizedTest 注解标记参数化测试方法，并指定测试名称模板
    @ValueSource(strings = {"N", "n", "No", "NO", "no"})  # 使用 ValueSource 注解提供参数化测试的参数
    public void testPromptBooleanAcceptsNo(String response) {  # 定义参数化测试方法，接收一个字符串参数
        // Given  # 标识测试的前置条件
        // 创建一个StringReader对象，用于读取response字符串的内容
        Reader in = new StringReader(response + "\n");
        // 创建一个StringWriter对象，用于将输出写入到字符串中
        StringWriter out = new StringWriter();
        // 创建一个UserIo对象，传入输入和输出流
        UserIo userIo = new UserIo(in, out);

        // 当
        // 调用userIo对象的promptBoolean方法，传入"TEST"作为提示信息
        boolean result = userIo.promptBoolean("TEST");

        // 然后
        // 断言输出的字符串应该是"TEST? "
        assertEquals("TEST? ", out.toString());
        // 断言result应该是false
        assertFalse(result);
    }

    @ParameterizedTest(name = "''{0}'' is accepted as ''yes''")
    // 使用@ValueSource注解，传入多个字符串作为参数化测试的输入
    @ValueSource(strings = {"Y", "y", "Yes", "YES", "yes", "", "foobar"})
    public void testPromptBooleanAcceptsYes(String response) {
        // 给定
        // 创建一个StringReader对象，用于读取response字符串的内容
        Reader in = new StringReader(response + "\n");
        // 创建一个StringWriter对象，用于将输出写入到字符串中
        StringWriter out = new StringWriter();
        // 创建一个UserIo对象，传入输入和输出流
        UserIo userIo = new UserIo(in, out);
        // 当
        boolean result = userIo.promptBoolean("TEST");

        // 然后
        assertEquals("TEST? ", out.toString());
        assertTrue(result);
    }

    @ParameterizedTest(name = "''{0}'' is accepted as number")
    @CsvSource({
        "1,1",
        "0,0",
        "-1,-1",
    })
    public void testPromptIntAcceptsNumbers(String response, int expected) {
        // 给定
        Reader in = new StringReader(response + "\n");
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);
        // 当
        int result = userIo.promptInt("TEST");

        // 预期
        assertEquals("TEST? ", out.toString());
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("promptInt should print an error and reprompt if given a non-numeric response")
    public void testPromptIntRepromptsOnNonNumeric() {
        // 给定
        Reader in = new StringReader("foo" + System.lineSeparator() +"1"); // 单词，然后数字
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);

        // 当
        int result = userIo.promptInt("TEST");

        // 预期
        # 断言输出字符串是否与预期相符
        assertEquals("TEST? !NUMBER EXPECTED - RETRY INPUT LINE" + System.lineSeparator() +"? ", out.toString())
        # 断言结果是否等于1
        assertEquals(1, result)
```