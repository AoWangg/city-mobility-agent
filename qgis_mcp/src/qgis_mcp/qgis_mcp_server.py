#!/usr/bin/env python3
"""
QGIS MCP Client - Simple client to connect to the QGIS MCP server
"""

import logging
from contextlib import asynccontextmanager
import socket
import json
from typing import AsyncIterator, Dict, Any
from mcp.server.fastmcp import FastMCP, Context

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QgisMCPServer")

class QgisMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """Connect to the QGIS MCP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"Successfully connected to QGIS MCP server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to server: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect from the server"""
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.info("Disconnected from QGIS MCP server.")

    def send_command(self, command_type, params=None):
        """Send a command to the server and get the response"""
        if not self.socket:
            logger.error("Not connected to server")
            return None

        # Create command
        command = {
            "type": command_type,
            "params": params or {}
        }

        try:
            # Send the command
            logger.debug(f"Sending command: {json.dumps(command)}")
            self.socket.sendall(json.dumps(command).encode('utf-8'))

            # Receive the response
            response_data = b''
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    # Server closed connection prematurely, or no more data
                    if not response_data: # No data received at all
                        logger.warning("No data received from server, connection might be closed.")
                        return None
                    break # Assume end of message if data was already received

                response_data += chunk

                # Try to decode as JSON to see if it's complete
                try:
                    # This check is crucial: ensure the full JSON message is received
                    decoded_response = json.loads(response_data.decode('utf-8'))
                    logger.debug(f"Received response: {decoded_response}")
                    return decoded_response
                except json.JSONDecodeError:
                    # Incomplete JSON, continue receiving
                    if len(response_data) > 1024 * 1024: # Safety break for very large non-JSON data
                        logger.error("Received very large data that is not valid JSON.")
                        return {"error": "Received very large non-JSON response", "data_preview": response_data[:200].decode('utf-8', errors='ignore')}
                    continue
            
            # If loop finished due to no chunk and we have partial data that isn't valid JSON
            logger.error(f"Failed to decode JSON response. Data: {response_data.decode('utf-8', errors='ignore')}")
            return {"error": "Failed to decode JSON response", "data_preview": response_data[:200].decode('utf-8', errors='ignore')}

        except socket.timeout:
            logger.error("Socket timeout during send/receive.")
            return {"error": "Socket timeout"}
        except ConnectionResetError:
            logger.error("Connection reset by server.")
            self.disconnect() # Ensure socket is cleaned up
            return {"error": "Connection reset by server"}
        except Exception as e:
            logger.error(f"Error sending/receiving command: {str(e)}")
            return {"error": f"Generic send/receive error: {str(e)}"}


_qgis_connection = None

def get_qgis_connection():
    """Get or create a persistent Qgis connection"""
    global _qgis_connection

    if _qgis_connection is not None:
        try:
            # A more robust way to check socket liveness for blocking sockets
            # is to attempt a non-blocking receive or a select call.
            # For simplicity in this client, we'll rely on send_command to detect dead sockets.
            # A better check would be a lightweight PING command if the server supports it without full command overhead.
            # For now, assume if object exists, it might still be good. Send_command will verify.
            # To truly test, you might send a specific "heartbeat" or "test_connection" command.
            # Example test (could fail if server expects JSON always):
            # _qgis_connection.socket.sendall(b'{"type":"ping_test"}') # Simple test
            # response = _qgis_connection.socket.recv(1024) # This part is tricky with blocking I/O
            logger.info("Reusing existing QGIS connection.")
            return _qgis_connection
        except (socket.error, AttributeError) as e: # AttributeError if socket is None
            logger.warning(f"Existing connection test failed or socket was None: {str(e)}")
            if _qgis_connection and _qgis_connection.socket:
                 try:
                    _qgis_connection.disconnect()
                 except Exception as disc_e:
                    logger.error(f"Error disconnecting old connection: {disc_e}")
            _qgis_connection = None

    if _qgis_connection is None:
        logger.info("No existing QGIS connection or connection was invalid. Creating new one.")
        _qgis_connection = QgisMCPServer(host="localhost", port=9876) # Ensure host/port are configurable if needed
        if not _qgis_connection.connect():
            logger.error("Failed to connect to QGIS. Make sure the QGIS plugin is running and listening.")
            _qgis_connection = None # Important to set to None on failure
            raise Exception("Could not connect to QGIS. Make sure the QGIS plugin server is running.")
        logger.info("Created new persistent connection to QGIS.")

    return _qgis_connection

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    logger.info("QgisMCPServer (FastMCP) server starting up...")
    try:
        # Attempt to establish the global QGIS connection on startup
        # to verify connectivity with the QGIS plugin.
        get_qgis_connection()
        logger.info("Successfully tested or established connection to QGIS plugin on startup.")
        yield {} # Context for the server if needed
    except Exception as e:
        logger.error(f"Failed to connect to QGIS plugin on startup: {str(e)}")
        logger.error("The FastMCP server will run, but QGIS tools will likely fail.")
        logger.error("Ensure the QGIS plugin providing the MCP service is running on localhost:9876.")
        yield {} # Still yield, but tools might not work
    finally:
        global _qgis_connection
        if _qgis_connection:
            logger.info("Disconnecting from QGIS on FastMCP server shutdown.")
            _qgis_connection.disconnect()
            _qgis_connection = None
        logger.info("QgisMCPServer (FastMCP) server shut down.")

mcp = FastMCP(
    "Qgis_mcp",
    description="QGIS integration through the Model Context Protocol",
    lifespan=server_lifespan
)

@mcp.tool()
def ping(ctx: Context) -> str:
    """Simple ping command to check server connectivity"""
    qgis = get_qgis_connection()
    result = qgis.send_command("ping")
    return json.dumps(result, indent=2)

@mcp.tool()
def get_qgis_info(ctx: Context) -> str:
    """Get QGIS information"""
    qgis = get_qgis_connection()
    result = qgis.send_command("get_qgis_info")
    return json.dumps(result, indent=2)

@mcp.tool()
def load_project(ctx: Context, path: str) -> str:
    """Load a QGIS project from the specified path."""
    qgis = get_qgis_connection()
    result = qgis.send_command("load_project", {"path": path})
    return json.dumps(result, indent=2)

@mcp.tool()
def create_new_project(ctx: Context, path: str) -> str:
    """Create a new project a save it"""
    qgis = get_qgis_connection()
    result = qgis.send_command("create_new_project", {"path": path})
    return json.dumps(result, indent=2)

@mcp.tool()
def get_project_info(ctx: Context) -> str:
    """Get current project information"""
    qgis = get_qgis_connection()
    result = qgis.send_command("get_project_info")
    return json.dumps(result, indent=2)

@mcp.tool()
def add_vector_layer(ctx: Context, path: str, provider: str = "ogr", name: str = None) -> str:
    """Add a vector layer to the project."""
    qgis = get_qgis_connection()
    params = {"path": path, "provider": provider}
    if name:
        params["name"] = name
    result = qgis.send_command("add_vector_layer", params)
    return json.dumps(result, indent=2)

@mcp.tool()
def add_raster_layer(ctx: Context, path: str, provider: str = "gdal", name: str = None) -> str:
    """Add a raster layer to the project."""
    qgis = get_qgis_connection()
    params = {"path": path, "provider": provider}
    if name:
        params["name"] = name
    result = qgis.send_command("add_raster_layer", params)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_layers(ctx: Context) -> str:
    """Retrieve all layers in the current project."""
    qgis = get_qgis_connection()
    result = qgis.send_command("get_layers")
    return json.dumps(result, indent=2)

@mcp.tool()
def remove_layer(ctx: Context, layer_id: str) -> str:
    """Remove a layer from the project by its ID."""
    qgis = get_qgis_connection()
    result = qgis.send_command("remove_layer", {"layer_id": layer_id})
    return json.dumps(result, indent=2)

@mcp.tool()
def zoom_to_layer(ctx: Context, layer_id: str) -> str:
    """Zoom to the extent of a specified layer."""
    qgis = get_qgis_connection()
    result = qgis.send_command("zoom_to_layer", {"layer_id": layer_id})
    return json.dumps(result, indent=2)

@mcp.tool()
def get_layer_features(ctx: Context, layer_id: str, limit: int = 10) -> str:
    """Retrieve features from a vector layer with an optional limit."""
    qgis = get_qgis_connection()
    result = qgis.send_command("get_layer_features", {"layer_id": layer_id, "limit": limit})
    return json.dumps(result, indent=2)

@mcp.tool()
def execute_processing(ctx: Context, algorithm: str, parameters: dict) -> str:
    """Execute a processing algorithm with the given parameters."""
    qgis = get_qgis_connection()
    result = qgis.send_command("execute_processing", {"algorithm": algorithm, "parameters": parameters})
    return json.dumps(result, indent=2)

@mcp.tool()
def save_project(ctx: Context, path: str = None) -> str:
    """Save the current project to the given path, or to the current project path if not specified."""
    qgis = get_qgis_connection()
    params = {}
    if path:
        params["path"] = path
    result = qgis.send_command("save_project", params)
    return json.dumps(result, indent=2)

@mcp.tool()
def render_map(ctx: Context, path: str, width: int = 800, height: int = 600) -> str:
    """Render the current map view to an image file with the specified dimensions."""
    qgis = get_qgis_connection()
    result = qgis.send_command("render_map", {"path": path, "width": width, "height": height})
    return json.dumps(result, indent=2)

@mcp.tool()
def execute_code(ctx: Context, code: str) -> str:
    """Execute arbitrary PyQGIS code provided as a string."""
    qgis = get_qgis_connection()
    result = qgis.send_command("execute_code", {"code": code})
    return json.dumps(result, indent=2)

# --- 新添加的工具 ---
@mcp.tool()
def plan_route(ctx: Context, start_lon: float, start_lat: float, end_lon: float, end_lat: float) -> str:
    """
    Plan a route between two geographic points using public transit.
    This command sends the coordinates to the QGIS plugin, which should
    have a handler for the 'plan_route' command that executes the
    actual routing logic.
    """
    qgis = get_qgis_connection()
    params = {
        "start_lon": start_lon,
        "start_lat": start_lat,
        "end_lon": end_lon,
        "end_lat": end_lat
    }
    logger.info(f"Sending 'plan_route' command with params: {params}")
    result = qgis.send_command("plan_route", params)
    # The result from the QGIS plugin (which executes your routing script)
    # should be a JSON-serializable dictionary or list.
    return json.dumps(result, indent=2)
# --- 结束新添加的工具 ---

def main():
    """Run the MCP server"""
    # Example: mcp.run(host="0.0.0.0", port=8000) # To make it accessible on network
    mcp.run() # Defaults to localhost:8000 usually for FastMCP

if __name__ == "__main__":
    main()