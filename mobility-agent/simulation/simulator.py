from typing import List, Dict, Any
import asyncio
from models.agent import MobilityAgent
from models.profile import PersonalProfile
from models.recorder import DecisionRecorder
from config.settings import API_CONFIG
import os
from dotenv import load_dotenv

load_dotenv()

class MobilitySimulator:
    """出行决策仿真器"""
    
    def __init__(self, recorder: DecisionRecorder = None):
        self.recorder = recorder if recorder else DecisionRecorder()
        
    def create_agent(self, profile: PersonalProfile) -> MobilityAgent:
        """创建智能体实例"""
        return MobilityAgent(
            profile=profile,
            gaode_map_key=os.getenv("GAODE_MAP_KEY"),
            model_name=os.getenv("MODEL_NAME"),
            api_key=os.getenv("API_KEY")
        )
        
    async def run_single_scenario(
        self,
        profile: PersonalProfile,
        query: str,
        weather: str,
        temperature: float
    ) -> Dict[str, Any]:
        """运行单个场景"""
        agent = self.create_agent(profile)
        result = await agent.plan_route(query, weather, temperature)
        self.recorder.add_record(result)
        return result
        
    async def run_batch_scenarios(
        self,
        profiles: List[PersonalProfile],
        queries: List[str],
        weather_conditions: List[tuple]
    ) -> List[Dict[str, Any]]:
        """批量运行多个场景"""
        results = []
        
        for weather, temp in weather_conditions:
            print(f"\n当前天气：{weather}，温度：{temp}度")
            print("-" * 50)
            
            for profile in profiles:
                for query in queries:
                    result = await self.run_single_scenario(
                        profile=profile,
                        query=query,
                        weather=weather,
                        temperature=temp
                    )
                    print(f"\n{result['profile'].occupation}的选择：")
                    print(result['result'])
                    print("-" * 30)
                    results.append(result)
                    
        return results 