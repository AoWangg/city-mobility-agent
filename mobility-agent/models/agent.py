from typing import Dict, Any
from datetime import datetime
import os
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import re

from models.profile import PersonalProfile
from mobility_mcp.mcp_client import MCPClient

load_dotenv()

class MobilityAgent:
    def __init__(
        self,
        profile: PersonalProfile,
        gaode_map_key: str = os.getenv("GAODE_MAP_KEY"),
        model_name: str = os.getenv("MODEL_NAME", "qwen-plus"),
        api_key: str = os.getenv("API_KEY")
    ):
        if not all([gaode_map_key, api_key]):
            raise ValueError("必需的环境变量未设置：GAODE_MAP_KEY 和 API_KEY")
            
        self.profile = profile
        self.mcp_client = MCPClient(gaode_map_key)
        self.llm = init_chat_model(
            model_name,
            api_key=api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            model_provider="openai"
        )
        
    def _create_prompt(self, weather: str = "晴朗", temperature: float = 25, traffic_conditions: str = "") -> ChatPromptTemplate:
        # 先获取格式化后的个人资料字符串
        profile_str = self.profile.to_prompt_string()
        
        # 创建完整的系统提示
        system_prompt = f"""
        {profile_str}
         
        现在是{weather}天，气温{temperature}度。
        {traffic_conditions}
        
        我是一个真实的个体，当遇到出行问题时，我会这样思考：
        
        1. 考虑问题背景：
           - 我遇到了什么问题？
           - 这个问题会如何影响我的日常出行？
           - 该问题是否涉及我原有出行方式和可能的新增出行方式的比较，如果涉及，我一定要调用高德地图的工具
        
        2. 分析出行行为：
           - 我要用高德地图查询可能的路线
           - 我要结合自己的情况评估每个选项
           - 我要考虑路线的可行性和适合度
           
        3. 提出我的意见：
           - 如果我觉得当前交通系统有什么不合理的地方，我总是会利用投诉工具来表达我的看法
        
        我会基于以上因素，一定要充分利用高德地图的能力，用第一人称详细说明我的思考过程，然后给出我的回答。
        在思考过程中，如果我觉得当前交通系统有什么不合理的地方，我总是会利用投诉工具来表达我的看法。            
        我会用"<Thought></Thought>"和"<Answer></Answer>"这种形式分别标注我的思考过程和最终的回答。
        """
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    async def plan_route(self, query: str, weather: str = "晴朗", temperature: float = 25, traffic_conditions: str = "") -> Dict[str, Any]:
        start_time = datetime.now()
        result = await self._execute_plan(query, weather, temperature, traffic_conditions)
        end_time = datetime.now()
        # 记录决策时间
        result.update({
            "query_time": start_time,
            "decision_duration": (end_time - start_time).total_seconds(),
            "weather": weather,
            "temperature": temperature,
            "query": query
        })
        
        return result

    async def _execute_plan(self, query: str, weather: str, temperature: float, traffic_conditions: str) -> Dict[str, Any]:
        async def execute_with_tools(tools):
            prompt = self._create_prompt(weather, temperature, traffic_conditions)
            
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
        
        # 提取工具调用信息
        tool_calls = []
        for step in agent_response.get("intermediate_steps", []):
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                    tool_calls.append({
                        'tool': action.tool,
                        'input': action.tool_input,
                        'output': observation
                    })
        
        # 使用更精确的提取方式
        thought_match = re.search(r'<Thought>(.*?)</Thought>', response_text, re.DOTALL)
        answer_match = re.search(r'<Answer>(.*?)</Answer>', response_text, re.DOTALL)
        
        if thought_match and answer_match:
            thought_process = thought_match.group(1).strip()
            final_decision = answer_match.group(1).strip()
        else:
            thought_process = "未能解析出思考过程"
            final_decision = response_text
        
        return {
            "status": "success",
            "result": response_text,
            "thought_process": thought_process,
            "final_decision": final_decision,
            "steps": len(agent_response.get("intermediate_steps", [])),
            "profile": self.profile,
            "tool_calls": tool_calls
        } 