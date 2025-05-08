from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import pandas as pd
from datetime import datetime
import os

class TravelPreference(Enum):
    COMFORT = "comfort"      # 注重舒适度
    ECONOMIC = "economic"    # 注重经济性
    SPEED = "speed"         # 注重速度
    FLEXIBLE = "flexible"   # 灵活多变

@dataclass
class PersonalProfile:
    age: int
    occupation: str
    travel_preference: TravelPreference
    has_car: bool = False
    has_bike: bool = False
    max_walking_distance: float = 1.0  # 单位：公里
    
    def to_prompt_string(self) -> str:
        return f"""
        你是一个{self.age}岁的{self.occupation}。
        你的出行偏好是{self.travel_preference.value}。
        你{'有' if self.has_car else '没有'}私家车。
        你{'有' if self.has_bike else '没有'}自行车。
        你能接受的最大步行距离是{self.max_walking_distance}公里。
        """

class MobilityAgent:
    def __init__(
        self,
        profile: PersonalProfile,
        gaode_map_key: str = "ba4a49acc350b56513915a3b2b2d5b8f",
        model_name: str = "qwen-plus",
        api_key: str = "sk-94b8a8c203764fd5ba6be83ed52a4a4c"
    ):
        self.profile = profile
        self.gaode_map_key = gaode_map_key
        self.llm = init_chat_model(
            model_name,
            api_key=api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            model_provider="openai"
        )
        
    def _create_prompt(self, weather: str = "晴朗", temperature: float = 25) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", f"""我是一个{self.profile.age}岁的{self.profile.occupation}。
            
            我的个人特征是：
            - 我的出行偏好是{self.profile.travel_preference.value}
            - 我{'有' if self.profile.has_car else '没有'}私家车
            - 我{'有' if self.profile.has_bike else '没有'}自行车
            - 我能接受的最大步行距离是{self.profile.max_walking_distance}公里
            
            现在是{weather}天，气温{temperature}度。
            
            作为一个真实的个体，我需要规划自己的出行路线。我会按照以下步骤进行思考：
            
            1. 考虑环境因素：
               - 天气状况对我的影响
               - 温度对我的体力和舒适度的影响
            
            2. 评估个人条件：
               - 我的年龄和身体状况
               - 我的职业特点和时间安排
               - 我可用的交通工具
               - 我的步行能力范围
            
            3. 权衡出行偏好：
               - 我对舒适度/经济性/速度的要求
               - 我的时间价值
               - 我的预算考虑
            
            4. 分析路线选择：
               - 使用高德地图查询可能的路线
               - 结合个人情况评估每个选项
               - 考虑路线的可行性和适合度
            
            我会基于以上因素，用第一人称详细说明我的思考过程，然后给出最终决定。
            
            请用"思考过程："和"最终决定："分别标注我的思考过程和最终选择的路线。
            """),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    async def plan_route(self, query: str, weather: str = "晴朗", temperature: float = 25) -> Dict[str, Any]:
        start_time = datetime.now()
        result = await self._execute_plan(query, weather, temperature)
        end_time = datetime.now()
        
        # 添加时间信息到结果中
        result.update({
            "query_time": start_time,
            "decision_duration": (end_time - start_time).total_seconds(),
            "weather": weather,
            "temperature": temperature,
            "query": query
        })
        
        return result

    async def _execute_plan(self, query: str, weather: str, temperature: float) -> Dict[str, Any]:
        async with MultiServerMCPClient({
            "search": {
                "url": f"https://mcp.amap.com/sse?key={self.gaode_map_key}",
                "transport": "sse",
            }
        }) as client:
            prompt = self._create_prompt(weather, temperature)
            tools = client.get_tools()
            
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=10,
                return_intermediate_steps=True,
                handle_parsing_errors=True
            )
            
            agent_response = await agent_executor.ainvoke({
                "input": query
            })
            
            # 解析响应中的思考过程和最终决定
            response_text = agent_response.get("output", "")
            thought_process = ""
            final_decision = ""
            
            if "思考过程：" in response_text and "最终决定：" in response_text:
                parts = response_text.split("最终决定：")
                thought_process = parts[0].replace("思考过程：", "").strip()
                final_decision = parts[1].strip()
            else:
                thought_process = "未能解析出思考过程"
                final_decision = response_text
            
            return {
                "status": "success",
                "result": response_text,
                "thought_process": thought_process,
                "final_decision": final_decision,
                "steps": len(agent_response.get("intermediate_steps", [])),
                "profile": self.profile
            }

class DecisionRecorder:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"mobility_decisions_{self.timestamp}.csv"
        self.filepath = os.path.join(os.path.dirname(__file__), "data", self.filename)
        self._initialize_csv()
    
    def _initialize_csv(self):
        headers = [
            'timestamp', 'decision_duration', 'weather', 'temperature', 
            'query', 'age', 'occupation', 'travel_preference', 'has_car', 
            'has_bike', 'max_walking_distance', 'decision_steps', 
            'thought_process', 'final_decision'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.filepath, index=False, encoding='utf-8')
        
    def add_record(self, decision_result: Dict[str, Any]):
        profile = decision_result['profile']
        record = {
            'timestamp': decision_result['query_time'],
            'decision_duration': decision_result['decision_duration'],
            'weather': decision_result['weather'],
            'temperature': decision_result['temperature'],
            'query': decision_result['query'],
            'age': profile.age,
            'occupation': profile.occupation,
            'travel_preference': profile.travel_preference.value,
            'has_car': profile.has_car,
            'has_bike': profile.has_bike,
            'max_walking_distance': profile.max_walking_distance,
            'decision_steps': decision_result['steps'],
            'thought_process': decision_result.get('thought_process', ''),
            'final_decision': decision_result.get('final_decision', '')
        }
        # 实时保存单条记录
        df = pd.DataFrame([record])
        df.to_csv(self.filepath, mode='a', header=False, index=False, encoding='utf-8')

async def simulate_different_agents():
    # 创建记录器
    recorder = DecisionRecorder()
    
    # 创建不同的个人档案
    profiles = [
        PersonalProfile(
            age=22,
            occupation="大学生",
            travel_preference=TravelPreference.ECONOMIC,
            has_bike=True,
            max_walking_distance=2.0
        ),
        PersonalProfile(
            age=35,
            occupation="上班族",
            travel_preference=TravelPreference.SPEED,
            has_car=True,
            max_walking_distance=1.0
        ),
        PersonalProfile(
            age=65,
            occupation="退休人员",
            travel_preference=TravelPreference.COMFORT,
            max_walking_distance=0.5
        ),
        PersonalProfile(
            age=28,
            occupation="外卖骑手",
            travel_preference=TravelPreference.FLEXIBLE,
            has_bike=True,
            max_walking_distance=3.0
        ),
        PersonalProfile(
            age=45,
            occupation="销售经理",
            travel_preference=TravelPreference.SPEED,
            has_car=True,
            max_walking_distance=1.5
        )
    ]
    
    agents = [MobilityAgent(profile) for profile in profiles]
    query = "我从同济大学嘉定校区到同济大学四平路校区，应该怎么走？"
    
    weather_conditions = [
        ("晴朗", 25),
        ("雨天", 20),
        ("炎热", 35)
    ]
    
    for weather, temp in weather_conditions:
        print(f"\n当前天气：{weather}，温度：{temp}度")
        print("-" * 50)
        
        for agent in agents:
            result = await agent.plan_route(query, weather, temp)
            print(f"\n{result['profile'].occupation}的选择：")
            print(result['result'])
            print("-" * 30)
            
            # 记录决策结果
            recorder.add_record(result)

if __name__ == "__main__":
    asyncio.run(simulate_different_agents()) 