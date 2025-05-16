from typing import List, Tuple
from models.profile import PersonalProfile, TravelPreference

def get_default_profiles() -> List[PersonalProfile]:
    """获取默认的个人档案列表"""
    return [
        PersonalProfile(
            age=22,
            occupation="大学生",
            travel_preference=TravelPreference.ECONOMIC,
            has_bike=True,
            max_walking_distance=2.0
        ),
        PersonalProfile(
            age=35,
            occupation="上班族",
            travel_preference=TravelPreference.SPEED,
            has_car=True,
            max_walking_distance=1.0
        ),
        PersonalProfile(
            age=65,
            occupation="退休人员",
            travel_preference=TravelPreference.COMFORT,
            max_walking_distance=0.5
        ),
        PersonalProfile(
            age=28,
            occupation="外卖骑手",
            travel_preference=TravelPreference.FLEXIBLE,
            has_bike=True,
            max_walking_distance=3.0
        ),
        PersonalProfile(
            age=45,
            occupation="销售经理",
            travel_preference=TravelPreference.SPEED,
            has_car=True,
            max_walking_distance=1.5
        )
    ]

def get_default_weather_conditions() -> List[Tuple[str, float]]:
    """获取默认的天气条件列表"""
    return [
        ("晴朗", 25),
        ("雨天", 20),
        ("炎热", 35)
    ]

class ScenarioBuilder:
    """场景构建器，用于自定义仿真场景"""
    
    def __init__(self):
        self.profiles = []
        self.weather_conditions = []
        self.queries = []
        self.traffic_conditions = []
    def add_profile(self, profile: PersonalProfile) -> 'ScenarioBuilder':
        self.profiles.append(profile)
        return self
        
    def add_weather_condition(self, weather: str, temperature: float) -> 'ScenarioBuilder':
        self.weather_conditions.append((weather, temperature))
        return self
    
    def add_traffic_condition(self, traffic: str) -> 'ScenarioBuilder':
        self.traffic_conditions.append(traffic)
        return self
        
    def add_query(self, query: str) -> 'ScenarioBuilder':
        self.queries.append(query)
        return self
        
    def build(self) -> dict:
        if not self.profiles:
            self.profiles = get_default_profiles()
        if not self.weather_conditions:
            self.weather_conditions = get_default_weather_conditions()
        if not self.queries:
            self.queries = [""]
            
        return {
            "profiles": self.profiles,
            "weather_conditions": self.weather_conditions,
            "queries": self.queries,
            "traffic_conditions": self.traffic_conditions
        } 