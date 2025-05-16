#!/usr/bin/env python3
"""
投诉意见处理服务 - 用于接收和记录用户投诉意见
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, Any
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ComplaintService")

# 定义投诉文件存储路径
COMPLAINT_FILE = Path("complaints.txt")

class ComplaintService:
    def __init__(self):
        """初始化投诉服务"""
        # 确保投诉文件存在
        if not COMPLAINT_FILE.exists():
            COMPLAINT_FILE.touch()
            logger.info(f"创建投诉记录文件: {COMPLAINT_FILE}")

    def save_complaint(self, content: str, contact: str = None) -> Dict[str, Any]:
        """
        保存投诉内容到文件
        
        Args:
            content: 投诉内容
            contact: 联系方式（可选）
            
        Returns:
            Dict: 包含操作结果的字典
        """
        try:
            # 准备投诉记录
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            complaint_record = {
                "timestamp": timestamp,
                "content": content,
                "contact": contact
            }
            
            # 将投诉记录写入文件
            with open(COMPLAINT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(complaint_record, ensure_ascii=False) + "\n")
            
            logger.info(f"成功记录投诉: {timestamp}")
            return {
                "status": "success",
                "message": "投诉已成功记录",
                "timestamp": timestamp
            }
            
        except Exception as e:
            error_msg = f"保存投诉记录时出错: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

    def get_complaints(self, limit: int = 10) -> Dict[str, Any]:
        """
        获取最近的投诉记录
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            Dict: 包含投诉记录的字典
        """
        try:
            complaints = []
            with open(COMPLAINT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        complaints.append(json.loads(line))
            
            # 按时间倒序排序并限制数量
            complaints.sort(key=lambda x: x["timestamp"], reverse=True)
            return {
                "status": "success",
                "complaints": complaints[:limit]
            }
            
        except Exception as e:
            error_msg = f"读取投诉记录时出错: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

# 创建投诉服务实例
complaint_service = ComplaintService()

# 先定义 server_lifespan 函数
@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """管理服务器启动和关闭的生命周期"""
    logger.info("投诉服务启动中...")
    try:
        yield {}
    finally:
        logger.info("投诉服务已关闭")

# 然后创建 MCP 服务实例
mcp = FastMCP(
    "complaint_service",
    description="投诉意见处理服务",
    lifespan=server_lifespan
)

@mcp.tool()
def submit_complaint(ctx: Context, content: str, contact: str = None) -> str:
    """
    提交新的投诉意见
    
    Args:
        content: 投诉内容
        contact: 联系方式（可选）
        
    Returns:
        str: JSON格式的操作结果
    """
    result = complaint_service.save_complaint(content, contact)
    return json.dumps(result, indent=2, ensure_ascii=False)

@mcp.tool()
def get_recent_complaints(ctx: Context, limit: int = 10) -> str:
    """
    获取最近的投诉记录
    
    Args:
        limit: 返回的最大记录数
        
    Returns:
        str: JSON格式的投诉记录列表
    """
    result = complaint_service.get_complaints(limit)
    return json.dumps(result, indent=2, ensure_ascii=False)

def main():
    """运行MCP服务器"""
    mcp.run()

if __name__ == "__main__":
    main()