# `.\pytorch\android\pytorch_android\src\androidTest\java\org\pytorch\suite\PytorchInstrumentedTestSuite.java`

```py
// 导入JUnit的RunWith和Suite类
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
// 导入PytorchInstrumentedTests类
import org.pytorch.PytorchInstrumentedTests;

// 使用@RunWith注解指定运行测试的方式为Suite运行器
@RunWith(Suite.class)
// 使用@SuiteClasses注解指定包含在测试套件中的测试类，这里只包含PytorchInstrumentedTests类
@Suite.SuiteClasses({PytorchInstrumentedTests.class})
// 定义PytorchInstrumentedTestSuite类，表示PyTorch仪表测试套件
public class PytorchInstrumentedTestSuite {}
```