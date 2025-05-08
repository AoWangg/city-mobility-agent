from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsDistanceArea,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsSpatialIndex,
    QgsFeatureRequest,
    QgsField
)
from qgis.PyQt.QtCore import QVariant
from qgis.utils import iface
import heapq
import collections
import math

BUS_LINES_LAYER_NAME = "上海市公交线路"
BUS_STOPS_LAYER_NAME = "上海市公交_点"

# --- 字段名配置 ---
STOP_PHYSICAL_ID_FIELD = None
STOP_NAME_FIELD_IN_STOPS = "Station"
LINE_NAME_FIELD_IN_STOPS = "BusName"
STOP_SEQUENCE_FIELD_IN_STOPS = "sequence"
STOP_LNG_FIELD = "Lng"
STOP_LAT_FIELD = "Lat"
STOP_DIR_NAME_FIELD_IN_STOPS = "Dir_Name"

LINE_NAME_FIELD_IN_LINES = "BusName"
LINE_DIR_NAME_FIELD_IN_LINES = "Dir_Name"

TRANSFER_PENALTY = 15
AVERAGE_BUS_SPEED_KMH = 20

# --- 2. 辅助函数 ---

def get_layer_by_name(layer_name):
    layers = QgsProject.instance().mapLayersByName(layer_name)
    if layers:
        return layers[0]
    else:
        print(f"错误：找不到图层 '{layer_name}'")
        return None

def find_nearest_physical_stop_id(point_geom, stops_layer, stops_info_dict):
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


def calculate_distance_meters(point1_geom, point2_geom):
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


def estimate_travel_time(stop_geom1, stop_geom2, average_bus_speed_kmh):
    if not stop_geom1 or not stop_geom2: return float('inf')
    distance_m = calculate_distance_meters(stop_geom1, stop_geom2)
    if distance_m == float('inf') or distance_m == 0:
        return 0.5

    distance_km = distance_m / 1000.0
    time_h = distance_km / average_bus_speed_kmh if average_bus_speed_kmh > 0 else float('inf')
    time_min = time_h * 60
    return max(0.5, time_min)

# --- 3. 构建公交网络图 ---
def build_transit_graph(stops_layer, lines_layer):
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
            
            time_estimate = estimate_travel_time(from_stop_geom, to_stop_geom, AVERAGE_BUS_SPEED_KMH)

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
def dijkstra_transit(graph, stops_info, start_physical_stop_id, end_physical_stop_id):
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


def format_path_output(path_details, stops_info):
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


def visualize_route_on_map(path_details, stops_info):
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
def plan_my_route(start_lon, start_lat, end_lon, end_lat):
    print("plan_my_route 函数开始执行...")
    # 步骤 0: 获取图层
    stops_layer = get_layer_by_name(BUS_STOPS_LAYER_NAME)
    lines_layer = get_layer_by_name(BUS_LINES_LAYER_NAME)

    if not stops_layer:
        print("错误：未能加载必要的公交站点图层。")
        return

    # 步骤 1: 构建公交网络图
    print("步骤 1: 构建公交网络图...")
    transit_graph, stops_data_dict = build_transit_graph(stops_layer, lines_layer)
    if not transit_graph or not stops_data_dict:
        print("公交网络图构建失败或为空。")
        return
    
    start_point_geom = QgsGeometry.fromPointXY(QgsPointXY(start_lon, start_lat))
    end_point_geom = QgsGeometry.fromPointXY(QgsPointXY(end_lon, end_lat))
    
    # 步骤 2: 查找最近的起终点物理站点
    print("步骤 2: 查找最近的起终点物理站点...")
    start_physical_id = find_nearest_physical_stop_id(start_point_geom, stops_layer, stops_data_dict)
    end_physical_id = find_nearest_physical_stop_id(end_point_geom, stops_layer, stops_data_dict)

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
    final_path_details = dijkstra_transit(transit_graph, stops_data_dict, start_physical_id, end_physical_id)

    # 步骤 4: 输出结果
    if final_path_details:
        print("\n--- 路径规划结果 ---")
        formatted_result = format_path_output(final_path_details, stops_data_dict)
        print(formatted_result)
        print("---------------------\n")
        
        # 步骤 5: 可视化路径
        print("步骤 5: 可视化路径...")
        visualize_route_on_map(final_path_details, stops_data_dict)
    else:
        print(f"未能找到从 {stops_data_dict[start_physical_id]['name']} 到 {stops_data_dict[end_physical_id]['name']} 的公交路径。")

    print("plan_my_route 函数执行完毕。")
    return final_path_details


print("脚本 route.py 开始执行...")

start_longitude = 121.4737
start_latitude = 31.2304
end_longitude = 121.4997
end_latitude = 31.2397

plan_my_route(start_longitude, start_latitude, end_longitude, end_latitude)

print("脚本 route.py 执行完毕。")