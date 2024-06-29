# `D:\src\scipysrc\matplotlib\doc\sphinxext\util.py`

```
import sys


# 导入sys模块，用于访问系统相关功能



def matplotlib_reduced_latex_scraper(block, block_vars, gallery_conf,
                                     **kwargs):
    """
    Reduce srcset when creating a PDF.

    Because sphinx-gallery runs *very* early, we cannot modify this even in the
    earliest builder-inited signal. Thus we do it at scraping time.
    """
    # 从sphinx_gallery.scrapers模块导入matplotlib_scraper函数
    from sphinx_gallery.scrapers import matplotlib_scraper

    # 如果当前构建器为latex，则清空gallery_conf中的'image_srcset'字段
    if gallery_conf['builder_name'] == 'latex':
        gallery_conf['image_srcset'] = []
    
    # 调用matplotlib_scraper函数处理block, block_vars, gallery_conf和kwargs，并返回结果
    return matplotlib_scraper(block, block_vars, gallery_conf, **kwargs)



# 清除basic_units模块，以便在导入时重新注册到单元注册表
def clear_basic_units(gallery_conf, fname):
    # 从sys.modules中移除'basic_units'模块的引用
    return sys.modules.pop('basic_units', None)
```