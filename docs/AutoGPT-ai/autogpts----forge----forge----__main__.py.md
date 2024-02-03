# `.\AutoGPT\autogpts\forge\forge\__main__.py`

```py
# 导入操作系统模块
import os

# 导入 uvicorn 模块
import uvicorn
# 导入 dotenv 模块中的 load_dotenv 函数
from dotenv import load_dotenv

# 导入 forge_log 模块
import forge.sdk.forge_log

# 创建 ForgeLogger 对象 LOG
LOG = forge.sdk.forge_log.ForgeLogger(__name__)

# 定义 logo 字符串
logo = """\n\n
       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
                                                                      
                                                                      
                                                                      
                8888888888                                            
                888                                                   
                888                                                   
                8888888  .d88b.  888d888 .d88b.   .d88b.              
                888     d88""88b 888P"  d88P"88b d8P  Y8b             
                888     888  888 888    888  888 88888888             
                888     Y88..88P 888    Y88b 888 Y8b.                 
                888      "Y88P"  888     "Y88888  "Y8888              
                                             888                      
                                        Y8b d88P                      
                                         "Y88P"                v0.1.0
\n"""

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 打印 logo 字符串
    print(logo)
    # 获取环境变量中的端口号，默认为 8000
    port = os.getenv("PORT", 8000)
    # 记录日志信息，显示 Agent 服务器启动的地址和端口号
    LOG.info(f"Agent server starting on http://localhost:{port}")
    # 加载环境变量
    load_dotenv()
    # 设置日志记录器
    forge.sdk.forge_log.setup_logger()
    # 运行uvicorn服务器，指定应用程序为"forge.app:app"
    uvicorn.run(
        "forge.app:app",
        # 指定主机为localhost
        host="localhost",
        # 将端口号转换为整数并指定端口号
        port=int(port),
        # 设置日志级别为error，只显示错误信息
        log_level="error",
        # 设置为True，使得在代码更改时重新加载应用程序
        reload=True,
    )
```