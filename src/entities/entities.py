from pydantic import BaseModel
from typing import List, Dict, Literal


class Coordinates(BaseModel):
    start: Dict[str, int]
    end: Dict[str, int]


class Defect(BaseModel):
    type: Literal['blur', 'abrasion', 'scratch', 'noise']
    coordinates: Coordinates
