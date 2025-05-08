from dataclasses import dataclass
from enum import Enum

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