from dataclasses import dataclass
from typing import Optional


@dataclass
class Quantity:
    symbol: str
    quantity_type: str
    value: float
    unit: str
    value_si: Optional[float] = None
