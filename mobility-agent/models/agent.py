from typing import Dict, Any
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

from models.profile import PersonalProfile
from models.mcp_client import MCPClient

class MobilityAgent:
    def __init__(
        self,
        profile: PersonalProfile,
        gaode_map_key: str = "ba4a49acc350b56513915a3b2b2d5b8f",
        model_name: str = "qwen-plus",
        api_key: str = "sk-94b8a8c203764fd5ba6be83ed52a4a4c"
    ):
        self.profile = profile
        self.mcp_client = MCPClient(gaode_map_key)
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
        async def execute_with_tools(tools):
            prompt = self._create_prompt(weather, temperature)
            
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
            
            return await agent_executor.ainvoke({
                "input": query
            })
            
        agent_response = await self.mcp_client.execute_with_tools(execute_with_tools)
        
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