from typing import Dict, Any, List
from langchain_mcp_adapters.client import MultiServerMCPClient

class MCPClient:
    def __init__(
        self,
        gaode_map_key: str,
        qgis_server_path: str = "/Users/aowang/code/city-mobility-agent/qgis_mcp/src/qgis_mcp",
        complaint_server_path: str = "/Users/aowang/code/city-mobility-agent/mobility-agent/mobility_mcp"  # 新增投诉服务路径
    ):
        self.gaode_map_key = gaode_map_key
        self.qgis_server_path = qgis_server_path
        self.complaint_server_path = complaint_server_path
        
    def _get_client_config(self) -> Dict[str, Any]:
        """获取MCP客户端配置"""
        return {
            "gmap": {
                "url": f"https://mcp.amap.com/sse?key={self.gaode_map_key}",
                "transport": "sse",
            },
            # "qgis": {
            #     "command": "uv",
            #     "args": [
            #         "--directory",
            #         self.qgis_server_path,
            #         "run",
            #         "qgis_mcp_server.py"
            #     ]
            # },
            "complaint": {  # 新增投诉服务配置
                "command": "python",
                "args": [
                    f"{self.complaint_server_path}/complaint_service.py"
                ]
            }
        }
    
    async def execute_with_tools(self, func) -> Any:
        """使用MCP工具执行函数"""
        async with MultiServerMCPClient(self._get_client_config()) as client:
            tools = client.get_tools()
            return await func(tools)
            
    @staticmethod
    def get_available_servers() -> List[str]:
        """获取可用的MCP服务器列表"""
        return ["gmap", "complaint"]  # 添加complaint服务