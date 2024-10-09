# `.\MinerU\magic_pdf\para\exceptions.py`

```
# 定义一个针对密集单行块的异常类
class DenseSingleLineBlockException(Exception):
    """
    该类定义了密集单行块的异常类型。
    """

    # 初始化异常类，设置默认消息
    def __init__(self, message="DenseSingleLineBlockException"):
        # 存储异常消息
        self.message = message
        # 调用父类初始化方法
        super().__init__(self.message)

    # 返回异常的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常的正式字符串表示
    def __repr__(self):
        return f"{self.message}"


# 定义一个针对标题检测的异常类
class TitleDetectionException(Exception):
    """
    该类定义了标题检测的异常类型。
    """

    # 初始化异常类，设置默认消息
    def __init__(self, message="TitleDetectionException"):
        # 存储异常消息
        self.message = message
        # 调用父类初始化方法
        super().__init__(self.message)

    # 返回异常的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常的正式字符串表示
    def __repr__(self):
        return f"{self.message}"


# 定义一个针对标题级别的异常类
class TitleLevelException(Exception):
    """
    该类定义了标题级别的异常类型。
    """

    # 初始化异常类，设置默认消息
    def __init__(self, message="TitleLevelException"):
        # 存储异常消息
        self.message = message
        # 调用父类初始化方法
        super().__init__(self.message)

    # 返回异常的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常的正式字符串表示
    def __repr__(self):
        return f"{self.message}"


# 定义一个针对段落拆分的异常类
class ParaSplitException(Exception):
    """
    该类定义了段落拆分的异常类型。
    """

    # 初始化异常类，设置默认消息
    def __init__(self, message="ParaSplitException"):
        # 存储异常消息
        self.message = message
        # 调用父类初始化方法
        super().__init__(self.message)

    # 返回异常的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常的正式字符串表示
    def __repr__(self):
        return f"{self.message}"


# 定义一个针对段落合并的异常类
class ParaMergeException(Exception):
    """
    该类定义了段落合并的异常类型。
    """

    # 初始化异常类，设置默认消息
    def __init__(self, message="ParaMergeException"):
        # 存储异常消息
        self.message = message
        # 调用父类初始化方法
        super().__init__(self.message)

    # 返回异常的字符串表示
    def __str__(self):
        return f"{self.message}"

    # 返回异常的正式字符串表示
    def __repr__(self):
        return f"{self.message}"


# 定义一个用于通过异常丢弃 PDF 文件的类
class DiscardByException:
    """
    该类通过异常丢弃 PDF 文件。
    """

    # 初始化方法，当前无操作
    def __init__(self) -> None:
        pass
    # 定义一个根据单行块异常丢弃 PDF 文件的函数
    def discard_by_single_line_block(self, pdf_dic, exception: DenseSingleLineBlockException):
        # 函数说明：根据单行块异常丢弃 PDF 文件，返回错误消息
        """
        This function discards pdf files by single line block exception
    
        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message
    
        Returns
        -------
        error_message : str
        """
        # 初始化异常页面计数器
        exception_page_nums = 0
        # 初始化页面计数器
        page_num = 0
        # 遍历 PDF 字典中的每一页
        for page_id, page in pdf_dic.items():
            # 检查页面 ID 是否以 "page_" 开头
            if page_id.startswith("page_"):
                # 增加页面计数
                page_num += 1
                # 检查页面是否包含预处理块
                if "preproc_blocks" in page.keys():
                    # 获取预处理块
                    preproc_blocks = page["preproc_blocks"]
    
                    # 初始化所有单行块的列表
                    all_single_line_blocks = []
                    # 遍历预处理块
                    for block in preproc_blocks:
                        # 检查块是否只有一行
                        if len(block["lines"]) == 1:
                            # 添加单行块到列表中
                            all_single_line_blocks.append(block)
    
                    # 检查预处理块和单行块的比例
                    if len(preproc_blocks) > 0 and len(all_single_line_blocks) / len(preproc_blocks) > 0.9:
                        # 如果比例大于 0.9，增加异常页面计数
                        exception_page_nums += 1
    
        # 如果没有页面，返回 None
        if page_num == 0:
            return None
    
        # 检查异常页面占比是否超过 0.1
        if exception_page_nums / page_num > 0.1:  # Low ratio means basically, whenever this is the case, it is discarded
            # 返回异常信息
            return exception.message
    
        # 返回 None，表示没有异常
        return None
    
    # 定义一个根据标题检测异常丢弃 PDF 文件的函数
    def discard_by_title_detection(self, pdf_dic, exception: TitleDetectionException):
        # 函数说明：根据标题检测异常丢弃 PDF 文件，返回错误消息
        """
        This function discards pdf files by title detection exception
    
        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message
    
        Returns
        -------
        error_message : str
        """
        # 返回 None，表示没有异常
        return None
    
    # 定义一个根据标题级别异常丢弃 PDF 文件的函数
    def discard_by_title_level(self, pdf_dic, exception: TitleLevelException):
        # 函数说明：根据标题级别异常丢弃 PDF 文件，返回错误消息
        """
        This function discards pdf files by title level exception
    
        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message
    
        Returns
        -------
        error_message : str
        """
        # 返回 None，表示没有异常
        return None
    
    # 定义一个根据段落分割异常丢弃 PDF 文件的函数
    def discard_by_split_para(self, pdf_dic, exception: ParaSplitException):
        # 函数说明：根据段落分割异常丢弃 PDF 文件，返回错误消息
        """
        This function discards pdf files by split para exception
    
        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message
    
        Returns
        -------
        error_message : str
        """
        # 返回 None，表示没有异常
        return None
    # 定义一个方法，接受 PDF 字典和合并参数异常作为参数
        def discard_by_merge_para(self, pdf_dic, exception: ParaMergeException):
            """
            该函数根据合并参数异常丢弃 PDF 文件
    
            参数
            ----------
            pdf_dic : dict
                PDF 字典
            exception : str
                异常信息
    
            返回
            -------
            error_message : str
            """
            # 返回异常的消息
            # return exception.message
            # 如果不返回任何信息，则返回 None
            return None
```