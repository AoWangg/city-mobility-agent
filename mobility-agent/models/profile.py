from dataclasses import dataclass
from enum import Enum

class TravelPreference(Enum):
    COMFORT = "comfort"      # 注重舒适度
    ECONOMIC = "economic"    # 注重经济性
    SPEED = "speed"         # 注重速度
    FLEXIBLE = "flexible"   # 灵活多变
    SAFE = "safe"           # 注重安全
@dataclass
class PersonalProfile:
    home_address_wgsx: float
    home_address_wgsy: float
    home_address_street: str
    home_address_district: str
    home_address_peoples: int
    home_address_children: int
    home_address_motorcycles: int
    home_address_cars: int
    age: int
    gender: str
    occupation: str
    travel_preference: TravelPreference
    has_car: bool = False
    has_bike: bool = False
    max_walking_distance: float = 1.0
    
    def to_prompt_string(self) -> str:
        return f"""
        ----------------我的基本情况----------------
        我住在{self.home_address_street}，具体位置在经纬度({self.home_address_wgsx}，{self.home_address_wgsy})，属于{self.home_address_street}街道，在{self.home_address_district}行政区。我家有{self.home_address_peoples}口人，其中有{self.home_address_children}个孩子。家里有{self.home_address_motorcycles}辆助动车和{self.home_address_cars}辆机动车。
        我今年{self.age}岁，是{self.gender}，我的职业是{self.occupation}。
    
        ----------------我的出行决策偏好----------------
        每次出行时，我会考虑以下三个核心因素：
        我的出行偏好（Attitudes）：我对不同出行方式（如地铁、公交、步行、骑行等）有什么样的感受和倾向？
        我感受到的社会影响（Subjective Norms）：我身边的亲友会推荐什么出行方式？我的选择会受到周围人的影响吗？
        我的实际能力（Perceived Behavioral Control）：我能轻松完成这种出行方式吗？我有足够的时间、体力和信息吗？
        我平时出行时比较注重{self.travel_preference.value}。
        我能接受的最远步行距离是{self.max_walking_distance}公里。      
        """ 