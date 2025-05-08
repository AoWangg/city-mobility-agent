from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from langchain.chat_models import init_chat_model
from langchain.agents import create_openai_functions_agent
 
llm_ds = init_chat_model(
    "qwen-plus",
    api_key = 'sk-94b8a8c203764fd5ba6be83ed52a4a4c',
    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model_provider="openai"
)
 
async def run_agent(query):
    gaode_map_key = "ba4a49acc350b56513915a3b2b2d5b8f"
    async with MultiServerMCPClient(
    {
        "gmap": {
            "url": f"https://mcp.amap.com/sse?key={gaode_map_key}",
            "transport": "sse",
        },
        "qgis": {
            "command": "uv",
            "args": [
                "--directory",
                "/Users/aowang/code/city-llm-mobility-predict/qgis_mcp/src/qgis_mcp",
                "run",
                "qgis_mcp_server.py"
            ]
        }
    }
) as client:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个上海市的居民，你擅长使用高德地图查询出行路线，并结合你的出现习惯和出行需求，思考最优的出行路线。"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        tools = client.get_tools()
        
        agent = create_openai_functions_agent(
            llm=llm_ds,
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
        
        # 运行代理
        agent_response = await agent_executor.ainvoke({
            "input": query
        })
        # 返回格式化的响应
        return {
            "status": "success",
            "result": agent_response.get("output", ""),
            "steps": len(agent_response.get("intermediate_steps", [])),
        }
 
print(asyncio.run(run_agent("我从同济大学嘉定校区到同济大学四平路校区应该怎么走，利用qgis做路径规划")))