from dataclasses import dataclass, asdict
import json


class UAVF_Dataclass:
    def to_string(self):
        return f"uavf:{json.dumps(asdict(self))}"
    @classmethod
    def from_string(cls, x):
        if x[:5] != 'uavf:':
            return None
        try:
            return cls(**json.loads(x[5:]))
        except:
            return None

@dataclass
class BumpLap(UAVF_Dataclass):
    lap_index: int

@dataclass
class RequestPayload(UAVF_Dataclass):
    shape_col: str
    letter_col: str
    letter: str
    shape: str

def is_uavf_message(message):
    return message[:5] == 'uavf:'
    

