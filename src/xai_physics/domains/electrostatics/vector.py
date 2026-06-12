from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def scale(self, factor: float) -> "Vec2":
        return Vec2(self.x * factor, self.y * factor)

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def unit(self) -> "Vec2":
        n = self.norm()
        if n == 0:
            raise ValueError("Zero vector has no unit direction.")
        return Vec2(self.x / n, self.y / n)
