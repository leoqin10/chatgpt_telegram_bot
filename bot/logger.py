import logging, os

# 定义日志文件的路径  
LOG_PATH = 'logs/app.log'
# 确保日志文件所在的目录存在  
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)   
# 创建一个logger  
Logger = logging.getLogger(__name__)  
Logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，这将记录所有DEBUG级别及以上的日志  
  
# 创建一个handler，用于写入日志文件  
file_handler = logging.FileHandler(LOG_PATH)  
file_handler.setLevel(logging.DEBUG)  # 设置handler的日志级别  
  
# 定义handler的输出格式  
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
file_handler.setFormatter(formatter)  
  
# 给logger添加handler  
Logger.addHandler(file_handler)  