import asyncio
from simulation.simulator import MobilitySimulator
from simulation.scenarios import ScenarioBuilder
from models.recorder import DecisionRecorder
from models.profile import PersonalProfile, TravelPreference

async def main():
    # 创建单个智能体的个人档案
    profile = PersonalProfile(
        age=28,
        occupation="同济大学学生",
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