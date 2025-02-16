from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
    text: List[str]
    label: str