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
    profile = PersonalProfile(
        home_address_wgsx=121.296123,
        home_address_wgsy=31.324903,
        home_address_street="同济大学嘉定校区",
        home_address_district="上海市嘉定区",
        home_address_peoples=3,
        home_address_children=1,
        home_address_motorcycles=1,
        home_address_cars=1,
        age=28,
        gender="男",
        occupation="青年教师，需要在同济大学两校区往返上课",
        travel_preference=TravelPreference.SPEED,
        has_car=False,
        max_walking_distance=1.0
    )
    
    scenario = ScenarioBuilder()\
        .add_profile(profile)\
        .add_weather_condition("晴朗", 25)\
        .add_traffic_condition("上海市新增了一条从绿苑路站到同济大学四平路校区的直达的公交线路")\
        .add_query(f"""
                   回答下列问卷，在最终的回答中给我完整的问卷答案：
                   """)\
        .build()
    
    simulator = MobilitySimulator(DecisionRecorder())
    
    results = await simulator.run_batch_scenarios(
        profiles=scenario["profiles"],
        queries=scenario["queries"],
        weather_conditions=scenario["weather_conditions"],
        traffic_conditions=scenario["traffic_conditions"]
    )
    
    print(f"\n仿真完成，共生成 {len(results)} 条决策记录")

if __name__ == "__main__":
    asyncio.run(main())
```

## 注意事项

- 请确保正确配置 API 密钥

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AoWangg/city-mobility-agent)