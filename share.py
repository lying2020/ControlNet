# 全局配置
save_memory = False

# 导入必要的模块
from cldm.hack import disable_verbosity, enable_sliced_attention

# 初始化设置
disable_verbosity()

if save_memory:
    enable_sliced_attention()
