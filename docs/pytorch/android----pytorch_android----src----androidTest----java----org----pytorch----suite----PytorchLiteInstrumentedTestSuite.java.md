# `.\pytorch\android\pytorch_android\src\androidTest\java\org\pytorch\suite\PytorchLiteInstrumentedTestSuite.java`

```py
# 导入所需的类库和模块
package org.pytorch.suite;

# 导入 JUnit 的运行器接口
import org.junit.runner.RunWith;

# 导入 JUnit 的套件注解
import org.junit.runners.Suite;

# 导入 PyTorch Lite 的仪器化测试类
import org.pytorch.PytorchLiteInstrumentedTests;

# 使用 Suite 运行器来运行测试套件
@RunWith(Suite.class)
# 指定套件的测试类，这里只包括 PytorchLiteInstrumentedTests 类
@Suite.SuiteClasses({PytorchLiteInstrumentedTests.class})
# 定义一个名为 PytorchLiteInstrumentedTestSuite 的测试套件类
public class PytorchLiteInstrumentedTestSuite {}
```