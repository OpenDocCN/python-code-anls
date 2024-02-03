# `.\PaddleOCR\test_tipc\web\index.test.js`

```py
# 导入 expect.json 文件中的数据
const expectData = require('./expect.json');

# 描述一个 e2e 测试 OCR 模型的测试套件
describe('e2e test ocr model', () => {

    # 在所有测试用例执行之前，打开指定路径的页面
    beforeAll(async () => {
        await page.goto(PATH);
    });

    # 结束 e2e 测试 OCR 模型的测试套件
    });
});
```