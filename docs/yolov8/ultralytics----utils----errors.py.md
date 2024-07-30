# `.\yolov8\ultralytics\utils\errors.py`

```py
# 导入从ultralytics.utils包中导入emojis函数
from ultralytics.utils import emojis

# 定义一个自定义异常类HUBModelError，用于处理Ultralytics YOLO模型获取相关的错误
class HUBModelError(Exception):
    """
    Custom exception class for handling errors related to model fetching in Ultralytics YOLO.
    
    This exception is raised when a requested model is not found or cannot be retrieved.
    The message is also processed to include emojis for better user experience.
    
    Attributes:
        message (str): The error message displayed when the exception is raised.
    
    Note:
        The message is automatically processed through the 'emojis' function from the 'ultralytics.utils' package.
    """
    
    def __init__(self, message="Model not found. Please check model URL and try again."):
        """Create an exception for when a model is not found."""
        # 调用父类的初始化方法
        super().__init__(emojis(message))
```