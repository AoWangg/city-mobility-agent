from typing import Dict, Any

# API配置
API_CONFIG = {
    "GAODE_MAP_KEY": "ba4a49acc350b56513915a3b2b2d5b8f",
    "MODEL_NAME": "qwen-plus",
    "API_KEY": "sk-94b8a8c203764fd5ba6be83ed52a4a4c",
    "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1"
}

# 默认天气条件
DEFAULT_WEATHER_CONDITIONS = [
    ("晴朗", 25),
    ("雨天", 20),
    ("炎热", 35)
]

# 默认查询
DEFAULT_QUERY = "我从同济大学嘉定校区到同济大学四平路校区，应该怎么走？"

# 智能体执行器配置
AGENT_EXECUTOR_CONFIG = {
    "verbose": True,
    "max_iterations": 10,
    "return_intermediate_steps": True,
    "handle_parsing_errors": True
}

# 数据存储配置
DATA_CONFIG = {
    "output_dir": "data",
    "csv_encoding": "utf-8"
} 