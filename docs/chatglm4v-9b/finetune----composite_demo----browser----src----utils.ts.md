# `.\chatglm4-finetune\composite_demo\browser\src\utils.ts`

```py
# 导入 winston 日志库
import winston from 'winston';

# 导入配置文件
import config from './config';

# 定义 TimeoutError 类，继承自 Error
export class TimeoutError extends Error {}

# 获取日志级别配置
const logLevel = config.LOG_LEVEL;

# 创建一个 logger 实例，用于记录日志
export const logger = winston.createLogger({
  # 设置日志级别
  level: logLevel,
  # 定义日志格式，包括颜色化和自定义输出
  format: winston.format.combine(
    winston.format.colorize(),
    winston.format.printf(info => {
      # 格式化日志信息输出，显示级别和消息
      return `${info.level}: ${info.message}`;
    }),
  ),
  # 定义日志传输方式，这里使用控制台输出
  transports: [new winston.transports.Console()],
});

# 在控制台输出当前日志级别
console.log('LOG_LEVEL', logLevel);

# 定义一个将高分辨率时间转换为毫秒的函数
export const parseHrtimeToMillisecond = (hrtime: [number, number]): number => {
    # 将高分辨率时间转换为毫秒
    return (hrtime[0] + hrtime[1] / 1e9) * 1000;
  };

# 定义一个封装 Promise 的函数，用于返回其值和执行时间
export const promiseWithTime = <T>(
    promise: Promise<T>
  ): Promise<{
    value: T;
    time: number;
  }> => {
    # 返回一个新的 Promise
    return new Promise((resolve, reject) => {
      # 记录开始时间
      const startTime = process.hrtime();
      # 处理传入的 Promise
      promise
        .then(value => {
          # 成功时解析，返回值和执行时间
          resolve({
            value: value,
            time: parseHrtimeToMillisecond(process.hrtime(startTime))
          });
        })
        .catch(err => reject(err)); # 捕获错误并拒绝
    });
  };

# 定义一个带超时功能的 Promise 函数
export const withTimeout = <T>(
    millis: number,
    promise: Promise<T>
  ): Promise<{
    value: T;
    time: number;
  }> => {
    # 创建一个超时的 Promise
    const timeout = new Promise<{ value: T; time: number }>((_, reject) =>
      # 指定时间后拒绝 Promise，抛出 TimeoutError
      setTimeout(() => reject(new TimeoutError()), millis)
    );
    # 竞争两个 Promise，哪个先完成就返回哪个
    return Promise.race([promiseWithTime(promise), timeout]);
  };
```