import asyncio
from simulation.simulator import MobilitySimulator
from simulation.scenarios import ScenarioBuilder
from models.recorder import DecisionRecorder
from models.profile import PersonalProfile, TravelPreference

async def main():
    profile = PersonalProfile(
        home_address_wgsx=121.296123,
        home_address_wgsy=31.324903,
        home_address_street="同济大学嘉定校区",
        home_address_district="上海市嘉定区",
        home_address_peoples=3,
        home_address_children=1,
        home_address_motorcycles=1,
        home_address_cars=1,
        age=28,
        gender="男",
        occupation="青年教师，需要在同济大学两校区往返上课",
        travel_preference=TravelPreference.SPEED,
        has_car=False,
        max_walking_distance=1.0
    )
    
    scenario = ScenarioBuilder()\
        .add_profile(profile)\
        .add_weather_condition("晴朗", 25)\
        .add_traffic_condition("上海市新增了一条从绿苑路站到同济大学四平路校区的直达的公交线路")\
        .add_query(f"""
                   回答下列问卷，在最终的回答中给我完整的问卷答案：
                   
                    公交线网调整对居民出行行为影响的调查问卷
                    一、背景信息
                    1. 您的性别：
                    ☐ 男   ☐ 女
                    2. 您的年龄：
                    ☐ 18岁以下 ☐ 18–25岁 ☐ 26–35岁 ☐ 36–45岁 ☐ 46岁以上
                    3. 您目前的职业：
                    ☐ 学生 ☐ 上班族（企业/政府/事业单位等） ☐ 自由职业者 ☐ 退休 ☐ 其他：______
                    4. 您的居住地址：________
                    5. 您平时最常使用的出行方式（可多选）：
                    ☐ 步行 ☐ 自行车 / 共享单车 ☐ 电动自行车 / 摩托车
                    ☐ 公共交通（公交车/地铁等） ☐ 网约车 / 出租车 ☐ 私家车 ☐ 其他：______
                    二、情景信息
                    1. 您是否了解近期的公交线路调整（用具体线路替换）：
                    ☐ 非常了解 ☐ 略有了解 ☐ 听说过，但不了解 ☐ 完全不了解
                    2. 您是否计划使用被调整的公交线路：
                    ☐ 计划经常使用 ☐ 计划偶尔使用 ☐ 计划不使用
                    3. 公交调整对您的出行路线是否产生影响：
                    ☐ 很大影响 ☐ 有一定影响 ☐ 基本无影响 ☐ 不清楚
                    三、出行行为信息
                    请结合以下假设行为作答：“在公交线路调整后，使用新线路或优化后的公交系统出行”。
                    请选择下列每一项在1至5之间的评分，其中：
                    1=完全不同意，2=不同意，3=一般，4=同意，5=非常同意。
                    A. 行为态度（Attitude）
                    1.我认为公交线网调整让出行体验变得更好。
                    2.调整后的公交线路更加合理，有助于提高出行效率。
                    3.使用新的公交线路是明智且高效的选择。
                    4.相比以前，现在乘坐公交让我感觉更舒适/便捷。
                    B. 主观规范（Subjective Norm）
                    5.我的家人或朋友建议我使用新的公交路线。
                    6.我的同事或周围人认为公交线网优化是积极的改变。
                    7.如果我使用新公交路线，身边人会认为我是明智/积极的。
                    8.我感觉大家都在逐渐适应和使用新线路。
                    C. 感知行为控制（Perceived Behavioral Control）
                    9.我了解新线路的走向与运行方式。
                    10.我能轻松找到新线路的信息（站点/时间/换乘等）。
                    11.我有能力根据公交调整重新规划出行。
                    12.即使原有线路发生改变，我也能顺利适应新路线。
                    D. 行为意图（Behavioral Intention）
                    13.我打算今后更多使用新的公交路线出行。
                    14.如果出行时间合适，我愿意选择新的公交替代原出行方式。
                    15.我已经调整了自己的通勤方式以适应公交线网变化。
                    四、实际行为变化
                    1. 公交调整后，您目前的通勤方式是否发生变化：
                    ☐ 没有变化 ☐ 增加了公交出行频率 ☐ 减少了公交出行频率
                    ☐ 改为骑行/步行 ☐ 改为私家车/网约车 ☐ 其他：______
                    2. 调整后您日常通勤所需时间变化为：
                    ☐ 明显缩短 ☐ 略微缩短 ☐ 无明显变化 ☐ 略微延长 ☐ 明显延长
                    五、开放式问题
                    1. 您如何评价此次公交线网调整？
                    2. 有哪些方面令您感到不便？
                    3. 您觉得怎样的优化措施可以进一步提升公交吸引力？
                   """)\
        .build()
    
    simulator = MobilitySimulator(DecisionRecorder())
    
    results = await simulator.run_batch_scenarios(
        profiles=scenario["profiles"],
        queries=scenario["queries"],
        weather_conditions=scenario["weather_conditions"],
        traffic_conditions=scenario["traffic_conditions"]
    )
    
    print(f"\n仿真完成，共生成 {len(results)} 条决策记录")

if __name__ == "__main__":
    asyncio.run(main())