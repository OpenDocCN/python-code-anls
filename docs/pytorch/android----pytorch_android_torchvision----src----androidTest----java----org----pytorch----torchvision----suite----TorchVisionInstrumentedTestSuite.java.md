# `.\pytorch\android\pytorch_android_torchvision\src\androidTest\java\org\pytorch\torchvision\suite\TorchVisionInstrumentedTestSuite.java`

```
// 导入 org.pytorch.torchvision.suite 包下的所有类
import org.pytorch.torchvision.suite;

// 导入 JUnit 框架中的 RunWith 类，用于指定测试运行器
import org.junit.runner.RunWith;

// 导入 JUnit 框架中的 Suite 类，用于指定测试套件
import org.junit.runners.Suite;

// 导入 TorchVisionInstrumentedTests 类，用于指定测试套件中包含的测试类
import org.pytorch.torchvision.TorchVisionInstrumentedTests;

// 指定该类使用 Suite 类作为测试运行器
@RunWith(Suite.class)

// 声明 TorchVisionInstrumentedTests 类为该测试套件中包含的测试类
@Suite.SuiteClasses({TorchVisionInstrumentedTests.class})
// 定义 TorchVisionInstrumentedTestSuite 类，表示 TorchVision 库的仪器化测试套件
public class TorchVisionInstrumentedTestSuite {}
```