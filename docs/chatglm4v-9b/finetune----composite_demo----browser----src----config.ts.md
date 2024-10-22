# `.\chatglm4-finetune\composite_demo\browser\src\config.ts`

```
# 导出一个默认的配置对象
export default {
    # 设置日志级别为 'debug'
    LOG_LEVEL: 'debug',
    # 设置浏览器超时时间为 10000 毫秒
    BROWSER_TIMEOUT: 10000,
    # 设置 Bing 搜索 API 的 URL
    BING_SEARCH_API_URL: 'https://api.bing.microsoft.com/v7.0/custom/',
    # 设置 Bing 搜索 API 的密钥
    BING_SEARCH_API_KEY: 'YOUR_BING_SEARCH_API_KEY',
    # 自定义配置 ID 的占位符，用户应在此处填入实际值
    CUSTOM_CONFIG_ID :  'YOUR_CUSTOM_CONFIG_ID', //将您的Custom Configuration ID放在此处
    # 设置主机地址为 'localhost'
    HOST: 'localhost',
    # 设置端口为 3000
    PORT: 3000,
};
```