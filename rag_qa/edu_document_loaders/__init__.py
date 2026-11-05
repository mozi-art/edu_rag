import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)


# 导入配置
from base.config import Config

# 添加配置中的路径到 sys.path
sys.path.append(Config().EDU_DOCUMENT_LOADERS_DIR)

# 条件导入各个文档加载器，允许在缺少依赖时继续运行
try:
    from .edu_docloader import *
except ImportError:
    pass

try:
    from .edu_pptloader import *
except ImportError:
    pass

try:
    from .edu_imgloader import *
except ImportError:
    pass

try:
    from .edu_pdfloader import *
except ImportError:
    pass

try:
    from .edu_csvloader import *
except ImportError:
    pass
except ImportError:
    pass