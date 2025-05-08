from mobility_agent import MobilityAgent, PersonalProfile, TravelPreference

# 创建用户画像
profile = PersonalProfile(
    age=30,
    occupation="上班族",
    travel_preference=TravelPreference.COMFORT,
    has_car=True,
    max_walking_distance=1.0
)

# 初始化出行助手
agent = MobilityAgent(profile)

# 规划路线
result =  agent.plan_route(
    query="从家到公司",
    weather="晴朗",
    temperature=25
)