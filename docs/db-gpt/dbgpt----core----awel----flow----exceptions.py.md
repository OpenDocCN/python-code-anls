# `.\DB-GPT-src\dbgpt\core\awel\flow\exceptions.py`

```py
"""定义 AWEL 流程中的异常类。"""


class FlowException(Exception):
    """AWEL 流程的基础异常类。"""

    def __init__(self, message: str, error_type: str = "Common Error"):
        """创建一个新的 FlowException 异常。"""
        super().__init__(message)
        self.message = message
        self.error_type = error_type


class FlowMetadataException(FlowException):
    """AWEL 流程元数据相关的异常基类。"""

    def __init__(self, message: str, error_type="build_metadata_error"):
        """创建一个新的 FlowMetadataException 异常。"""
        super().__init__(message, error_type)


class FlowParameterMetadataException(FlowMetadataException):
    """AWEL 流程参数元数据异常。"""

    def __init__(self, message: str, error_type="build_parameter_metadata_error"):
        """创建一个新的 FlowParameterMetadataException 异常。"""
        super().__init__(message, error_type)


class FlowClassMetadataException(FlowMetadataException):
    """AWEL 流程类元数据异常。

    当从元数据加载类失败时抛出。
    """

    def __init__(self, message: str, error_type="load_class_metadata_error"):
        """创建一个新的 FlowClassMetadataException 异常。"""
        super().__init__(message, error_type)


class FlowDAGMetadataException(FlowMetadataException):
    """从元数据构建 DAG 失败时抛出的异常。"""

    def __init__(self, message: str, error_type="build_dag_metadata_error"):
        """创建一个新的 FlowDAGMetadataException 异常。"""
        super().__init__(message, error_type)
```