import asyncio
from simulation.simulator import MobilitySimulator
from simulation.scenarios import ScenarioBuilder
from models.recorder import DecisionRecorder

async def main():
    # 创建场景构建器
    scenario = ScenarioBuilder().build()
    
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