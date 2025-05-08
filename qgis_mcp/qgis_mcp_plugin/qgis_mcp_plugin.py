import os
import json
import socket
import traceback
import collections # Added for routing
import heapq     # Added for routing
import math      # Added for routing

from qgis.core import *
from qgis.gui import *
from qgis.PyQt.QtCore import QObject, pyqtSignal, QTimer, Qt, QSize, QVariant
from qgis.PyQt.QtWidgets import QAction, QDockWidget, QVBoxLayout, QLabel, QPushButton, QSpinBox, QWidget
from qgis.PyQt.QtGui import QIcon, QColor # QColor was missing, needed for render_map
from qgis.utils import active_plugins, iface # iface can be used directly if module-level functions need it and it's passed

# --- Routing Constants (Define these according to your project) ---
BUS_LINES_LAYER_NAME = "上海市公交线路"
BUS_STOPS_LAYER_NAME = "上海市公交_点"

# --- Field name configurations ---
STOP_NAME_FIELD_IN_STOPS = "Station"
LINE_NAME_FIELD_IN_STOPS = "BusName"
STOP_SEQUENCE_FIELD_IN_STOPS = "sequence"
STOP_LNG_FIELD = "Lng"
STOP_LAT_FIELD = "Lat"
STOP_DIR_NAME_FIELD_IN_STOPS = "Dir_Name"

LINE_NAME_FIELD_IN_LINES = "BusName"
LINE_DIR_NAME_FIELD_IN_LINES = "Dir_Name"

TRANSFER_PENALTY = 15  # Minutes
AVERAGE_BUS_SPEED_KMH = 20 # km/h

class QgisMCPServer(QObject):
    """Server class to handle socket connections and execute QGIS commands"""
    
    def __init__(self, host='localhost', port=9876, iface=None):
        super().__init__()
        self.host = host
        self.port = port
        self.iface = iface
        self.running = False
        self.socket = None
        self.client = None
        self.buffer = b''
        self.timer = None
    
    def start(self):
        """Start the server"""
        self.running = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.setblocking(False)
            
            # Create a timer to process server operations
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_server)
            self.timer.start(100)  # 100ms interval
            
            QgsMessageLog.logMessage(f"QGIS MCP server started on {self.host}:{self.port}", "QGIS MCP")
            return True
        except Exception as e:
            QgsMessageLog.logMessage(f"Failed to start server: {str(e)}", "QGIS MCP", Qgis.Critical)
            self.stop()
            return False
            
    def stop(self):
        """Stop the server"""
        self.running = False
        
        if self.timer:
            self.timer.stop()
            self.timer = None
            
        if self.socket:
            self.socket.close()
        if self.client:
            self.client.close()
            
        self.socket = None
        self.client = None
        QgsMessageLog.logMessage("QGIS MCP server stopped", "QGIS MCP")
    
    def process_server(self):
        """Process server operations (called by timer)"""
        if not self.running:
            return
            
        try:
            # Accept new connections
            if not self.client and self.socket:
                try:
                    self.client, address = self.socket.accept()
                    self.client.setblocking(False)
                    QgsMessageLog.logMessage(f"Connected to client: {address}", "QGIS MCP")
                except BlockingIOError:
                    pass  # No connection waiting
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error accepting connection: {str(e)}", "QGIS MCP", Qgis.Warning)
                
            # Process existing connection
            if self.client:
                try:
                    # Try to receive data
                    try:
                        data = self.client.recv(8192)
                        if data:
                            self.buffer += data
                            # Try to process complete messages
                            try:
                                # Attempt to parse the buffer as JSON
                                command = json.loads(self.buffer.decode('utf-8'))
                                # If successful, clear the buffer and process command
                                self.buffer = b''
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                self.client.sendall(response_json.encode('utf-8'))
                            except json.JSONDecodeError:
                                # Incomplete data, keep in buffer
                                pass
                        else:
                            # Connection closed by client
                            QgsMessageLog.logMessage("Client disconnected", "QGIS MCP")
                            self.client.close()
                            self.client = None
                            self.buffer = b''
                    except BlockingIOError:
                        pass  # No data available
                    except Exception as e:
                        QgsMessageLog.logMessage(f"Error receiving data: {str(e)}", "QGIS MCP", Qgis.Warning)
                        self.client.close()
                        self.client = None
                        self.buffer = b''
                        
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error with client: {str(e)}", "QGIS MCP", Qgis.Warning)
                    if self.client:
                        self.client.close()
                        self.client = None
                    self.buffer = b''
                    
        except Exception as e:
            QgsMessageLog.logMessage(f"Server error: {str(e)}", "QGIS MCP", Qgis.Critical)

    def execute_command(self, command):
        """Execute a command"""
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            
            handlers = {
                "ping": self.ping,
                "get_qgis_info": self.get_qgis_info,
                "load_project": self.load_project,
                "get_project_info": self.get_project_info,
                "execute_code": self.execute_code,
                "add_vector_layer": self.add_vector_layer,
                "add_raster_layer": self.add_raster_layer,
                "get_layers": self.get_layers,
                "remove_layer": self.remove_layer,
                "zoom_to_layer": self.zoom_to_layer,
                "get_layer_features": self.get_layer_features,
                "execute_processing": self.execute_processing,
                "save_project": self.save_project,
                "render_map": self.render_map,
                "create_new_project": self.create_new_project,
                "plan_route": self.plan_route,
            }
            
            handler = handlers.get(cmd_type)
            if handler:
                try:
                    QgsMessageLog.logMessage(f"Executing handler for {cmd_type}", "QGIS MCP")
                    result = handler(**params)
                    QgsMessageLog.logMessage(f"Handler execution complete", "QGIS MCP")
                    return {"status": "success", "result": result}
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error in handler: {str(e)}", "QGIS MCP", Qgis.Critical)
                    traceback.print_exc()
                    return {"status": "error", "message": str(e)}
            else:
                return {"status": "error", "message": f"Unknown command type: {cmd_type}"}
                
        except Exception as e:
            QgsMessageLog.logMessage(f"Error executing command: {str(e)}", "QGIS MCP", Qgis.Critical)
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    # Command handlers
    def ping(self, **kwargs):
        """Simple ping command"""
        return {"pong": True}
    
    def get_qgis_info(self, **kwargs):
        """Get basic QGIS information"""
        return {
            "qgis_version": Qgis.version(),
            "profile_folder": QgsApplication.qgisSettingsDirPath(),
            "plugins_count": len(active_plugins)
        }
    
    def get_project_info(self, **kwargs):
        """Get information about the current QGIS project"""
        project = QgsProject.instance()
        
        # Get basic project information
        info = {
            "filename": project.fileName(),
            "title": project.title(),
            "layer_count": len(project.mapLayers()),
            "crs": project.crs().authid(),
            "layers": []
        }
        
        # Add basic layer information (limit to 10 layers for performance)
        layers = list(project.mapLayers().values())
        for i, layer in enumerate(layers):
            if i >= 10:  # Limit to 10 layers
                break
                
            layer_info = {
                "id": layer.id(),
                "name": layer.name(),
                "type": self._get_layer_type(layer),
                "visible": layer.isValid() and project.layerTreeRoot().findLayer(layer.id()).isVisible()
            }
            info["layers"].append(layer_info)
        
        return info
    
    def _get_layer_type(self, layer):
        """Helper to get layer type as string"""
        if layer.type() == QgsMapLayer.VectorLayer:
            return f"vector_{layer.geometryType()}"
        elif layer.type() == QgsMapLayer.RasterLayer:
            return "raster"
        else:
            return str(layer.type())
    
    def execute_code(self, code, **kwargs):
        """Execute arbitrary PyQGIS code"""
        try:
            # Create a local namespace for execution
            namespace = {
                "qgis": Qgis,
                "QgsProject": QgsProject,
                "iface": self.iface,
                "QgsApplication": QgsApplication,
                "QgsVectorLayer": QgsVectorLayer,
                "QgsRasterLayer": QgsRasterLayer,
                "QgsCoordinateReferenceSystem": QgsCoordinateReferenceSystem
            }
            
            # Execute the code
            exec(code, namespace)
            return {"executed": True}
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")
    
    def get_layer_by_name(self, layer_name):
        layers = QgsProject.instance().mapLayersByName(layer_name)
        if layers:
            return layers[0]
        else:
            print(f"错误：找不到图层 '{layer_name}'")
            return None

    def find_nearest_physical_stop_id(self, point_geom, stops_layer, stops_info_dict):
        if not stops_layer or not point_geom or not stops_info_dict:
            return None

        min_dist = float('inf')
        nearest_physical_id = None
        
        for physical_id, info in stops_info_dict.items():
            stop_geom = info['geom']
            if not stop_geom or stop_geom.isEmpty():
                continue
            
            current_dist = point_geom.distance(stop_geom)
            if current_dist < min_dist:
                min_dist = current_dist
                nearest_physical_id = physical_id
                
        if nearest_physical_id:
            print(f"找到最近物理站点ID: {nearest_physical_id} (名称: {stops_info_dict[nearest_physical_id]['name']}), 距离: {min_dist:.2f} 地图单位")
        else:
            print("警告: 未能从 stops_info_dict 找到最近的物理站点。")
            
        return nearest_physical_id

    def calculate_distance_meters(self, point1_geom, point2_geom):
        if not point1_geom or not point2_geom or point1_geom.isEmpty() or point2_geom.isEmpty():
            return float('inf')

        p1 = point1_geom.asPoint()
        p2 = point2_geom.asPoint()

        distance_calculator = QgsDistanceArea()
        distance_calculator.setEllipsoid(QgsProject.instance().ellipsoid())

        qgs_point1 = QgsPointXY(p1.x(), p1.y())
        qgs_point2 = QgsPointXY(p2.x(), p2.y())
        
        try:
            dist_meters = distance_calculator.measureLine(qgs_point1, qgs_point2)
            return dist_meters
        except Exception as e:
            R = 6371000
            lat1_rad = math.radians(p1.y())
            lon1_rad = math.radians(p1.x())
            lat2_rad = math.radians(p2.y())
            lon2_rad = math.radians(p2.x())
            dlon = lon2_rad - lon1_rad
            dlat = lat2_rad - lat1_rad
            a_hav = (math.sin(dlat / 2)**2 +
                    math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2)
            c_hav = 2 * math.atan2(math.sqrt(a_hav), math.sqrt(1 - a_hav))
            return R * c_hav

    def estimate_travel_time(self, stop_geom1, stop_geom2, average_bus_speed_kmh):
        if not stop_geom1 or not stop_geom2: return float('inf')
        distance_m = self.calculate_distance_meters(stop_geom1, stop_geom2)
        if distance_m == float('inf') or distance_m == 0:
            return 0.5

        distance_km = distance_m / 1000.0
        time_h = distance_km / average_bus_speed_kmh if average_bus_speed_kmh > 0 else float('inf')
        time_min = time_h * 60
        return max(0.5, time_min)

    # --- 3. 构建公交网络图 ---
    def build_transit_graph(self, stops_layer, lines_layer):
        print("开始构建公交网络图...")
        graph = collections.defaultdict(lambda: collections.defaultdict(list))
        stops_info = {}

        if not stops_layer or not lines_layer:
            print("错误：公交站点或线路图层未加载。")
            return {}, {}

        print("  步骤 3.1: 收集物理站点信息...")
        physical_stop_locations = {}
        _physical_id_counter = 0
        temp_stops_data_by_qgis_fid = {}

        for stop_feature in stops_layer.getFeatures():
            try:
                qgis_fid = stop_feature.id()
                stop_name = stop_feature[STOP_NAME_FIELD_IN_STOPS]
                line_name = stop_feature[LINE_NAME_FIELD_IN_STOPS]
                dir_name = stop_feature[STOP_DIR_NAME_FIELD_IN_STOPS]
                sequence = stop_feature[STOP_SEQUENCE_FIELD_IN_STOPS]
                lng = stop_feature[STOP_LNG_FIELD]
                lat = stop_feature[STOP_LAT_FIELD]

                if any(v is None for v in [stop_name, line_name, sequence, lng, lat]):
                    continue
                
                loc_key = f"{lng}_{lat}"
                
                current_physical_id = -1
                if loc_key not in physical_stop_locations:
                    _physical_id_counter += 1
                    physical_stop_locations[loc_key] = _physical_id_counter
                    current_physical_id = _physical_id_counter
                    
                    stops_info[current_physical_id] = {
                        'name': stop_name,
                        'geom': QgsGeometry.fromPointXY(QgsPointXY(lng, lat)),
                        'qgis_fids_at_loc': {qgis_fid},
                        'lines_served': set()
                    }
                else:
                    current_physical_id = physical_stop_locations[loc_key]
                    stops_info[current_physical_id]['qgis_fids_at_loc'].add(qgis_fid)

                temp_stops_data_by_qgis_fid[qgis_fid] = {
                    'physical_id': current_physical_id,
                    'line': str(line_name).strip(),
                    'dir': str(dir_name).strip() if dir_name else None,
                    'seq': int(sequence),
                    'name': str(stop_name).strip()
                }
                stops_info[current_physical_id]['lines_served'].add(str(line_name).strip())

            except Exception as e:
                print(f"处理站点要素 {stop_feature.id()} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"  物理站点识别完毕。共 {len(stops_info)} 个。")
        if not stops_info:
            print("错误：未能从站点图层提取任何有效的物理站点信息。")
            return {}, {}

        print("  步骤 3.2: 构建线路站点序列...")
        line_stop_sequences = collections.defaultdict(list)
        for qgis_fid, data in temp_stops_data_by_qgis_fid.items():
            line_key = (data['line'], data['dir'])
            line_stop_sequences[line_key].append((data['seq'], data['physical_id'], data['name']))

        for line_key in line_stop_sequences:
            line_stop_sequences[line_key].sort(key=lambda x: x[0])

        print("  步骤 3.3: 构建图的边...")
        for line_key, sorted_stops_on_line in line_stop_sequences.items():
            line_name, dir_name = line_key
            if len(sorted_stops_on_line) < 2:
                continue

            for i in range(len(sorted_stops_on_line) - 1):
                seq1, from_physical_stop_id, name1 = sorted_stops_on_line[i]
                seq2, to_physical_stop_id, name2 = sorted_stops_on_line[i+1]

                if from_physical_stop_id not in stops_info or to_physical_stop_id not in stops_info:
                    print(f"警告: 线路 {line_key} 上的站点ID {from_physical_stop_id} 或 {to_physical_stop_id} 未在stops_info中找到。")
                    continue

                from_stop_geom = stops_info[from_physical_stop_id]['geom']
                to_stop_geom = stops_info[to_physical_stop_id]['geom']
                
                time_estimate = self.estimate_travel_time(from_stop_geom, to_stop_geom, AVERAGE_BUS_SPEED_KMH)

                existing_edge = False
                for edge_data in graph[from_physical_stop_id][to_physical_stop_id]:
                    if edge_data['line'] == line_name and edge_data.get('dir') == dir_name:
                        existing_edge = True
                        break
                if not existing_edge:
                    graph[from_physical_stop_id][to_physical_stop_id].append({
                        'line': line_name,
                        'time': time_estimate,
                        'dir': dir_name
                    })

        if not graph:
            print("错误: 未能构建任何图的边。检查站点序列和数据。")
        print(f"公交网络图构建完成。物理站点数: {len(stops_info)}, 图中含边的起点数: {len(graph)}")
        return graph, stops_info

    # --- 4. Dijkstra 算法 (考虑换乘) ---
    def dijkstra_transit(self, graph, stops_info, start_physical_stop_id, end_physical_stop_id):
        print(f"开始路径规划：从物理站点 {start_physical_stop_id} ({stops_info.get(start_physical_stop_id, {}).get('name', '未知起点')}) 到 {end_physical_stop_id} ({stops_info.get(end_physical_stop_id, {}).get('name', '未知终点')})")

        pq = []
        start_stop_name = stops_info.get(start_physical_stop_id, {}).get('name', '未知起点')
        initial_path_segment = (start_physical_stop_id, None, f"从 {start_stop_name} 出发", 0, None)
        heapq.heappush(pq, (0, start_physical_stop_id, None, [initial_path_segment]))

        visited_costs = {}
        found_path = None

        while pq:
            cost, current_stop_id, current_line_taken, path = heapq.heappop(pq)
            
            current_stop_name = stops_info.get(current_stop_id, {}).get('name', f"ID:{current_stop_id}")

            if (current_stop_id, current_line_taken) in visited_costs and \
            visited_costs[(current_stop_id, current_line_taken)] <= cost:
                continue
            visited_costs[(current_stop_id, current_line_taken)] = cost
            
            if current_stop_id == end_physical_stop_id:
                print(f"已找到到达终点 {current_stop_name} 的路径！成本: {cost:.2f}")
                if found_path is None or cost < found_path[0][0]:
                    found_path = (path, cost)
                return path # Return first found path

            if current_stop_id in graph:
                for neighbor_stop_id, connections in graph[current_stop_id].items():
                    for conn in connections:
                        edge_line = conn['line']
                        edge_time = conn['time']
                        edge_dir = conn.get('dir')

                        new_segment_cost = edge_time
                        is_transfer = False
                        
                        if current_line_taken is not None and current_line_taken != edge_line:
                            new_segment_cost += TRANSFER_PENALTY
                            is_transfer = True
                        
                        new_total_cost = cost + new_segment_cost
                        
                        neighbor_stop_name = stops_info.get(neighbor_stop_id, {}).get('name', f"ID:{neighbor_stop_id}")
                        action_desc = ""
                        if is_transfer:
                            action_desc = f"在 {current_stop_name} 换乘 {edge_line}"
                            if edge_dir: action_desc += f" ({edge_dir} 方向)"
                            action_desc += f", 前往 {neighbor_stop_name}"
                        else:
                            action_desc = f"乘坐 {edge_line}"
                            if edge_dir: action_desc += f" ({edge_dir} 方向)"
                            action_desc += f", 前往 {neighbor_stop_name}"

                        if (neighbor_stop_id, edge_line) not in visited_costs or \
                        new_total_cost < visited_costs.get((neighbor_stop_id, edge_line), float('inf')):
                            
                            new_path_segment = (neighbor_stop_id, edge_line, action_desc, new_total_cost, edge_dir)
                            heapq.heappush(pq, (new_total_cost, neighbor_stop_id, edge_line, path + [new_path_segment]))
        
        if found_path:
            return found_path[0]

        print("未能找到路径。")
        return None

    def format_path_output(self, path_details, stops_info):
        if not path_details:
            return "未能找到路径。"

        output_lines = ["公交路径规划结果:"]
        
        start_stop_id, _, start_action, _, _ = path_details[0]
        start_stop_name = stops_info.get(start_stop_id, {}).get('name', f"ID:{start_stop_id}")
        output_lines.append(f"1. {start_action} (站点: {start_stop_name})")

        current_乘坐_line = None
        current_乘坐_direction = None
        segment_start_stop_name = start_stop_name
        intermediate_stops_on_segment = []

        for i in range(1, len(path_details)):
            to_stop_id, line_taken_for_this_leg, action_desc, _, dir_for_this_leg = path_details[i]
            to_stop_name = stops_info.get(to_stop_id, {}).get('name', f"ID:{to_stop_id}")
            
            if line_taken_for_this_leg != current_乘坐_line or dir_for_this_leg != current_乘坐_direction :
                if current_乘坐_line and intermediate_stops_on_segment:
                    output_lines.append(f"   途经: {', '.join(intermediate_stops_on_segment)}")
                
                line_info_str = f"{line_taken_for_this_leg}"
                if dir_for_this_leg:
                    line_info_str += f" ({dir_for_this_leg} 方向)"

                if path_details[i-1][1] is not None and (line_taken_for_this_leg != path_details[i-1][1] or dir_for_this_leg != path_details[i-1][4]):
                    prev_stop_name_for_transfer = stops_info.get(path_details[i-1][0], {}).get('name', f"ID:{path_details[i-1][0]}")
                    output_lines.append(f"{len(output_lines)}. 在 {prev_stop_name_for_transfer} 换乘 {line_info_str}")
                else:
                    output_lines.append(f"{len(output_lines)}. 乘坐 {line_info_str}")

                current_乘坐_line = line_taken_for_this_leg
                current_乘坐_direction = dir_for_this_leg
                segment_start_stop_name = stops_info.get(path_details[i-1][0], {}).get('name', f"ID:{path_details[i-1][0]}")
                intermediate_stops_on_segment = []

            if i > 0 and to_stop_name != segment_start_stop_name :
                if i < len(path_details) -1 :
                    intermediate_stops_on_segment.append(to_stop_name)

        if current_乘坐_line and intermediate_stops_on_segment:
            output_lines.append(f"   途经: {', '.join(intermediate_stops_on_segment)}")
        
        final_stop_id = path_details[-1][0]
        final_stop_name = stops_info.get(final_stop_id, {}).get('name', f"ID:{final_stop_id}")
        output_lines.append(f"   在 {final_stop_name} 到达目的地。")
        
        return "\n".join(output_lines)

    def visualize_route_on_map(self, path_details, stops_info):
        if not path_details or not iface or not stops_info:
            print("无法可视化路径：缺少路径详情、QGIS界面或站点信息。")
            return

        vl_name = "规划路径"
        layers = QgsProject.instance().mapLayersByName(vl_name)
        if layers:
            for layer_to_remove in layers:
                QgsProject.instance().removeMapLayer(layer_to_remove.id())

        # Assume WGS84 (EPSG:4326) for the layer CRS, as Lng/Lat data is based on this.
        crs_str = "EPSG:4326"

        vl = QgsVectorLayer(f"LineString?crs={crs_str}", vl_name, "memory")
        if not vl.isValid():
            print(f"错误: 创建内存图层 '{vl_name}' 失败。")
            return
            
        pr = vl.dataProvider()

        pr.addAttributes([
            QgsField("leg", QVariant.Int),
            QgsField("line", QVariant.String),
            QgsField("direction", QVariant.String),
            QgsField("action", QVariant.String)
        ])
        vl.updateFields()

        for i in range(len(path_details) - 1):
            from_physical_stop_id = path_details[i][0]
            to_physical_stop_id = path_details[i+1][0]
            
            line_name = path_details[i+1][1] if path_details[i+1][1] else "步行至站点"
            action = path_details[i+1][2]
            direction_val = path_details[i+1][4] if len(path_details[i+1]) > 4 and path_details[i+1][4] else ""

            if from_physical_stop_id in stops_info and to_physical_stop_id in stops_info:
                from_geom_obj = stops_info[from_physical_stop_id].get('geom')
                to_geom_obj = stops_info[to_physical_stop_id].get('geom')

                if from_geom_obj and not from_geom_obj.isEmpty() and \
                to_geom_obj and not to_geom_obj.isEmpty():
                    try:
                        from_point = from_geom_obj.asPoint()
                        to_point = to_geom_obj.asPoint()
                        
                        line_segment = QgsGeometry.fromPolylineXY([from_point, to_point])
                        if line_segment.isEmpty():
                            print(f"警告: 创建空的线段，从 {from_physical_stop_id} 到 {to_physical_stop_id}")
                            continue

                        feat = QgsFeature()
                        feat.setGeometry(line_segment)
                        feat.setAttributes([i + 1, str(line_name), str(direction_val), str(action)])
                        pr.addFeature(feat)
                    except Exception as e_geom:
                        print(f"警告: 创建线段或要素时出错 (从 {from_physical_stop_id} 到 {to_physical_stop_id}): {e_geom}")
                else:
                    print(f"警告: 可视化时，站点ID {from_physical_stop_id} 或 {to_physical_stop_id} 的几何信息为空或无效。")
            else:
                print(f"警告: 可视化时，站点ID {from_physical_stop_id} 或 {to_physical_stop_id} 的信息未在 stops_info 中找到。")

        vl.updateExtents()
        QgsProject.instance().addMapLayer(vl)
        if vl.isValid():
            print(f"路径已添加到地图图层 '{vl_name}'。")
        else:
            print(f"错误: 添加图层 '{vl_name}' 到地图后，图层变为无效。")

    # --- 5. 主执行函数 ---
    def plan_my_route(self, start_lon, start_lat, end_lon, end_lat):
        print("plan_my_route 函数开始执行...")
        # 步骤 0: 获取图层
        stops_layer = self.get_layer_by_name(BUS_STOPS_LAYER_NAME)
        lines_layer = self.get_layer_by_name(BUS_LINES_LAYER_NAME)

        if not stops_layer:
            print("错误：未能加载必要的公交站点图层。")
            return

        # 步骤 1: 构建公交网络图
        print("步骤 1: 构建公交网络图...")
        transit_graph, stops_data_dict = self.build_transit_graph(stops_layer, lines_layer)
        if not transit_graph or not stops_data_dict:
            print("公交网络图构建失败或为空。")
            return
        
        start_point_geom = QgsGeometry.fromPointXY(QgsPointXY(start_lon, start_lat))
        end_point_geom = QgsGeometry.fromPointXY(QgsPointXY(end_lon, end_lat))
        
        # 步骤 2: 查找最近的起终点物理站点
        print("步骤 2: 查找最近的起终点物理站点...")
        start_physical_id = self.find_nearest_physical_stop_id(start_point_geom, stops_layer, stops_data_dict)
        end_physical_id = self.find_nearest_physical_stop_id(end_point_geom, stops_layer, stops_data_dict)

        if not start_physical_id or not end_physical_id:
            print("错误：未能找到起点或终点附近的物理公交站点。")
            if not start_physical_id: print(f"  - 起点 ({start_lon},{start_lat}) 未匹配到站点。")
            if not end_physical_id: print(f"  - 终点 ({end_lon},{end_lat}) 未匹配到站点。")
            return

        print(f"起点匹配到物理站点: {stops_data_dict[start_physical_id]['name']} (ID: {start_physical_id})")
        print(f"终点匹配到物理站点: {stops_data_dict[end_physical_id]['name']} (ID: {end_physical_id})")

        if start_physical_id == end_physical_id:
            print("起点和终点是同一个物理站点。无需规划。")
            return "起点和终点为同一站点。"

        # 步骤 3: 执行路径规划
        print(f"步骤 3: 执行Dijkstra路径规划从 {start_physical_id} 到 {end_physical_id}...")
        final_path_details = self.dijkstra_transit(transit_graph, stops_data_dict, start_physical_id, end_physical_id)

        # 步骤 4: 输出结果
        if final_path_details:
            print("\n--- 路径规划结果 ---")
            formatted_result = self.format_path_output(final_path_details, stops_data_dict)
            print(formatted_result)
            print("---------------------\n")
            
            # 步骤 5: 可视化路径
            print("步骤 5: 可视化路径...")
            self.visualize_route_on_map(final_path_details, stops_data_dict)
        else:
            print(f"未能找到从 {stops_data_dict[start_physical_id]['name']} 到 {stops_data_dict[end_physical_id]['name']} 的公交路径。")

        print("plan_my_route 函数执行完毕。")
        return final_path_details
    
    def plan_route(self, start_lon, start_lat, end_lon, end_lat, **kwargs):
        """规划从起点到终点的公交路线"""
        try:
            # 调用路径规划函数
            path_details = self.plan_my_route(start_lon, start_lat, end_lon, end_lat)
            
            if not path_details:
                return {
                    "status": "error",
                    "message": "未能找到可行的公交路线"
                }
            
            # 获取图层
            stops_layer = self.get_layer_by_name(BUS_STOPS_LAYER_NAME)
            if not stops_layer:
                return {
                    "status": "error",
                    "message": "无法获取公交站点图层"
                }
            
            # 构建站点信息字典
            stops_info = {}
            for feature in stops_layer.getFeatures():
                stop_id = feature.id()
                stops_info[stop_id] = {
                    'name': feature[STOP_NAME_FIELD_IN_STOPS],
                    'geom': feature.geometry()
                }
            
            # 格式化路径输出
            formatted_result = self.format_path_output(path_details, stops_info)
            
            # 可视化路径
            self.visualize_route_on_map(path_details, stops_info)
            
            # 返回完整的路径信息
            return {
                "status": "success",
                "result": {
                    "path_details": path_details,
                    "formatted_route": formatted_result,
                    "stops_info": {
                        stop_id: {
                            'name': info['name'],
                            'geometry': info['geom'].asWkt() if info['geom'] else None
                        }
                        for stop_id, info in stops_info.items()
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"路径规划过程中发生错误: {str(e)}"
            }
    
    def add_vector_layer(self, path, name=None, provider="ogr", **kwargs):
        """Add a vector layer to the project"""
        if not name:
            name = os.path.basename(path)
            
        # Create the layer
        layer = QgsVectorLayer(path, name, provider)
        
        if not layer.isValid():
            raise Exception(f"Layer is not valid: {path}")
        
        # Add to project
        QgsProject.instance().addMapLayer(layer)
        
        return {
            "id": layer.id(),
            "name": layer.name(),
            "type": self._get_layer_type(layer),
            "feature_count": layer.featureCount()
        }
    
    def add_raster_layer(self, path, name=None, provider="gdal", **kwargs):
        """Add a raster layer to the project"""
        if not name:
            name = os.path.basename(path)
            
        # Create the layer
        layer = QgsRasterLayer(path, name, provider)
        
        if not layer.isValid():
            raise Exception(f"Layer is not valid: {path}")
        
        # Add to project
        QgsProject.instance().addMapLayer(layer)
        
        return {
            "id": layer.id(),
            "name": layer.name(),
            "type": "raster",
            "width": layer.width(),
            "height": layer.height()
        }
    
    def get_layers(self, **kwargs):
        """Get all layers in the project"""
        project = QgsProject.instance()
        layers = []
        
        for layer_id, layer in project.mapLayers().items():
            layer_info = {
                "id": layer_id,
                "name": layer.name(),
                "type": self._get_layer_type(layer),
                "visible": project.layerTreeRoot().findLayer(layer_id).isVisible()
            }
            
            # Add type-specific information
            if layer.type() == QgsMapLayer.VectorLayer:
                layer_info.update({
                    "feature_count": layer.featureCount(),
                    "geometry_type": layer.geometryType()
                })
            elif layer.type() == QgsMapLayer.RasterLayer:
                layer_info.update({
                    "width": layer.width(),
                    "height": layer.height()
                })
                
            layers.append(layer_info)
        
        return layers
    
    def remove_layer(self, layer_id, **kwargs):
        """Remove a layer from the project"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            project.removeMapLayer(layer_id)
            return {"removed": layer_id}
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def zoom_to_layer(self, layer_id, **kwargs):
        """Zoom to a layer's extent"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            layer = project.mapLayer(layer_id)
            self.iface.setActiveLayer(layer)
            self.iface.zoomToActiveLayer()
            return {"zoomed_to": layer_id}
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def get_layer_features(self, layer_id, limit=10, **kwargs):
        """Get features from a vector layer"""
        project = QgsProject.instance()
        
        if layer_id in project.mapLayers():
            layer = project.mapLayer(layer_id)
            
            if layer.type() != QgsMapLayer.VectorLayer:
                raise Exception(f"Layer is not a vector layer: {layer_id}")
            
            features = []
            for i, feature in enumerate(layer.getFeatures()):
                if i >= limit:
                    break
                    
                # Extract attributes
                attrs = {}
                for field in layer.fields():
                    attrs[field.name()] = feature.attribute(field.name())
                
                # Extract geometry if available
                geom = None
                if feature.hasGeometry():
                    geom = {
                        "type": feature.geometry().type(),
                        "wkt": feature.geometry().asWkt(precision=4)
                    }
                
                features.append({
                    "id": feature.id(),
                    "attributes": attrs,
                    "geometry": geom
                })
            
            return {
                "layer_id": layer_id,
                "feature_count": layer.featureCount(),
                "features": features,
                "fields": [field.name() for field in layer.fields()]
            }
        else:
            raise Exception(f"Layer not found: {layer_id}")
    
    def execute_processing(self, algorithm, parameters, **kwargs):
        """Execute a processing algorithm"""
        try:
            import processing
            result = processing.run(algorithm, parameters)
            return {
                "algorithm": algorithm,
                "result": {k: str(v) for k, v in result.items()}  # Convert values to strings for JSON
            }
        except Exception as e:
            raise Exception(f"Processing error: {str(e)}")
    
    def save_project(self, path=None, **kwargs):
        """Save the current project"""
        project = QgsProject.instance()
        
        if not path and not project.fileName():
            raise Exception("No project path specified and no current project path")
        
        save_path = path if path else project.fileName()
        if project.write(save_path):
            return {"saved": save_path}
        else:
            raise Exception(f"Failed to save project to {save_path}")
    
    def load_project(self, path, **kwargs):
        """Load a project"""
        project = QgsProject.instance()
        
        if project.read(path):
            self.iface.mapCanvas().refresh()
            return {
                "loaded": path,
                "layer_count": len(project.mapLayers())
            }
        else:
            raise Exception(f"Failed to load project from {path}")
    
    def create_new_project(self, path, **kwargs):
        """
        Creates a new QGIS project and saves it at the specified path.
        If a project is already loaded, it clears it before creating the new one.
        
        :param project_path: Full path where the project will be saved
                            (e.g., 'C:/path/to/project.qgz')
        """
        project = QgsProject.instance()
        
        if project.fileName():
            project.clear()
        
        project.setFileName(path)
        self.iface.mapCanvas().refresh()
        
        # Save the project
        if project.write():
            return {
                "created": f"Project created and saved successfully at: {path}",
                "layer_count": len(project.mapLayers())
            }
        else:
            raise Exception(f"Failed to save project to {path}")
    
    def render_map(self, path, width=800, height=600, **kwargs):
        """Render the current map view to an image"""
        try:
            # Create map settings
            ms = QgsMapSettings()
            
            # Set layers to render
            layers = list(QgsProject.instance().mapLayers().values())
            ms.setLayers(layers)
            
            # Set map canvas properties
            rect = self.iface.mapCanvas().extent()
            ms.setExtent(rect)
            ms.setOutputSize(QSize(width, height))
            ms.setBackgroundColor(QColor(255, 255, 255))
            ms.setOutputDpi(96)
            
            # Create the render
            render = QgsMapRendererParallelJob(ms)
            
            # Start rendering
            render.start()
            render.waitForFinished()
            
            # Get the image and save
            img = render.renderedImage()
            if img.save(path):
                return {
                    "rendered": True,
                    "path": path,
                    "width": width,
                    "height": height
                }
            else:
                raise Exception(f"Failed to save rendered image to {path}")
                
        except Exception as e:
            raise Exception(f"Render error: {str(e)}")


class QgisMCPDockWidget(QDockWidget):
    """Dock widget for the QGIS MCP plugin"""
    closed = pyqtSignal()
    
    def __init__(self, iface):
        super().__init__("QGIS MCP")
        self.iface = iface
        self.server = None
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the dock widget UI"""
        # Create widget and layout
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Add port selection
        layout.addWidget(QLabel("Server Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setMinimum(1024)
        self.port_spin.setMaximum(65535)
        self.port_spin.setValue(9876)
        layout.addWidget(self.port_spin)
        
        # Add server control buttons
        self.start_button = QPushButton("Start Server")
        self.start_button.clicked.connect(self.start_server)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Add status label
        self.status_label = QLabel("Server: Stopped")
        layout.addWidget(self.status_label)
        
        # Add to dock widget
        self.setWidget(widget)
    
    def start_server(self):
        """Start the server"""
        if not self.server:
            port = self.port_spin.value()
            self.server = QgisMCPServer(port=port, iface=self.iface)
            
        if self.server.start():
            self.status_label.setText(f"Server: Running on port {self.server.port}")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.port_spin.setEnabled(False)
    
    def stop_server(self):
        """Stop the server"""
        if self.server:
            self.server.stop()
            self.server = None
            
        self.status_label.setText("Server: Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.port_spin.setEnabled(True)
        
    def closeEvent(self, event):
        """Stop server on dock close"""
        self.stop_server()
        self.closed.emit()
        super().closeEvent(event)


class QgisMCPPlugin:
    """Main plugin class for QGIS MCP"""
    
    def __init__(self, iface):
        self.iface = iface
        self.dock_widget = None
        self.action = None
    
    def initGui(self):
        """Initialize GUI"""
        # Create action
        self.action = QAction(
            "QGIS MCP",
            self.iface.mainWindow()
        )
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock)
        
        # Add to plugins menu and toolbar
        self.iface.addPluginToMenu("QGIS MCP", self.action)
        self.iface.addToolBarIcon(self.action)
    
    def toggle_dock(self, checked):
        """Toggle the dock widget"""
        if checked:
            # Create dock widget if it doesn't exist
            if not self.dock_widget:
                self.dock_widget = QgisMCPDockWidget(self.iface)
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
                # Connect close event
                self.dock_widget.closed.connect(self.dock_closed)
            else:
                # Show existing dock widget
                self.dock_widget.show()
        else:
            # Hide dock widget
            if self.dock_widget:
                self.dock_widget.hide()
    
    def dock_closed(self):
        """Handle dock widget closed"""
        self.action.setChecked(False)
    
    def unload(self):
        """Unload plugin"""
        # Stop server if running
        if self.dock_widget:
            self.dock_widget.stop_server()
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget = None
            
        # Remove plugin menu item and toolbar icon
        self.iface.removePluginMenu("QGIS MCP", self.action)
        self.iface.removeToolBarIcon(self.action)


# Plugin entry point
def classFactory(iface):
    return QgisMCPPlugin(iface)
