# 城市出行智能体

这是一个基于大模型和高德地图MCP的城市出行智能体

## 主要功能

- 个性化出行规划：考虑用户年龄、职业、出行偏好等个人特征
- 多维度决策：综合考虑天气、温度、交通方式等因素
- 智能路线推荐：基于高德地图API提供最优路线
- 决策记录与分析：记录用户的出行决策过程，支持后续分析

## 技术特点

- 使用 LangChain 框架构建智能代理
- 集成高德地图 API 进行路线规划
- 采用异步编程提高性能
- 支持多种出行偏好（舒适型、经济型、速度型、灵活型）
- 完整的决策记录和分析功能

## 环境要求

- Python 3.11
- 依赖包见 `requirements.txt`

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
- 需要配置高德地图 API Key
- 需要配置 AI 模型 API Key

3. 运行示例：
```bash
cd mobility-agent
python main.py
```

## 项目结构

- `mobility-agent/`: 核心代码目录
- `try_mobility_agent.py`: 示例运行脚本
- `requirements.txt`: 项目依赖
- `qgis_mcp/`: QGIS相关功能模块

## 单智能体使用示例(mobility-agent/one_agent.py)

```python
import asyncio
from simulation.simulator import MobilitySimulator
from simulation.scenarios import ScenarioBuilder
from models.recorder import DecisionRecorder
from models.profile import PersonalProfile, TravelPreference

async def main():
    # 创建单个智能体的个人档案
    profile = PersonalProfile(
        age=28,
        occupation="上班族",
        travel_preference=TravelPreference.SPEED,
        has_car=True,
        max_walking_distance=1.0
    )
    
    # 创建场景构建器并添加单个智能体
    scenario = ScenarioBuilder()\
        .add_profile(profile)\
        .add_weather_condition("晴朗", 25)\
        .add_query("我从同济大学嘉定校区到同济大学四平路校区，应该怎么走？")\
        .build()
    
    # 创建仿真器
    simulator = MobilitySimulator(DecisionRecorder())
    
    # 运行仿真
    results = await simulator.run_batch_scenarios(
        profiles=scenario["profiles"],
        queries=scenario["queries"],
        weather_conditions=scenario["weather_conditions"]
    )
    
    print(f"\n仿真完成，共生成 {len(results)} 条决策记录")

if __name__ == "__main__":
    asyncio.run(main())
```

## 注意事项

- 请确保正确配置 API 密钥