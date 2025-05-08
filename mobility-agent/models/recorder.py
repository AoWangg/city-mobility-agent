from typing import Dict, Any
import os
import pandas as pd
from datetime import datetime

class DecisionRecorder:
    def __init__(self, output_dir: str = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"mobility_decisions_{self.timestamp}.csv"
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, self.filename)
        self._initialize_csv()
    
    def _initialize_csv(self):
        headers = [
            'timestamp', 'decision_duration', 'weather', 'temperature', 
            'query', 'age', 'occupation', 'travel_preference', 'has_car', 
            'has_bike', 'max_walking_distance', 'decision_steps', 
            'thought_process', 'final_decision'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.filepath, index=False, encoding='utf-8')
        
    def add_record(self, decision_result: Dict[str, Any]):
        profile = decision_result['profile']
        record = {
            'timestamp': decision_result['query_time'],
            'decision_duration': decision_result['decision_duration'],
            'weather': decision_result['weather'],
            'temperature': decision_result['temperature'],
            'query': decision_result['query'],
            'age': profile.age,
            'occupation': profile.occupation,
            'travel_preference': profile.travel_preference.value,
            'has_car': profile.has_car,
            'has_bike': profile.has_bike,
            'max_walking_distance': profile.max_walking_distance,
            'decision_steps': decision_result['steps'],
            'thought_process': decision_result.get('thought_process', ''),
            'final_decision': decision_result.get('final_decision', '')
        }
        # 实时保存单条记录
        df = pd.DataFrame([record])
        df.to_csv(self.filepath, mode='a', header=False, index=False, encoding='utf-8') 