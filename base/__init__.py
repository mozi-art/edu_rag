import os,sys
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,project_dir)


from base.logger import logger
from base.config import config
