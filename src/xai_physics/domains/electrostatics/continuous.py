from __future__ import annotations

import math
from typing import Any

from xai_physics.core.answer import AnswerEnvelope
from xai_physics.core.result import SolveResult
from xai_physics.core.units import to_si

K_COULOMB = 9.0e9
EPS0 = 8.85e-12


class ContinuousDistributionError(ValueError):
    pass


def _q(obj: dict[str, Any], *, default_unit: str | None = None) -> float:
    if obj is None:
        raise ContinuousDistributionError("missing quantity")
    value = obj.get("value")
    unit = obj.get("unit") or default_unit or ""
    if value is None:
        raise ContinuousDistributionError(f"quantity has no value: {obj!r}")
    unit = str(unit).replace("^2", "²")
    # Density units are not part of the generic unit table because they are not
    # base scalar answer units.  Convert them locally.
    if unit in {"C/m2", "C/m^2", "C/m²"}:
        return float(value)
    if unit in {"uC/m2", "uC/m^2", "uC/m²", "μC/m²"}:
        return float(value) * 1e-6
    if unit in {"C/m"}:
        return float(value)
    if unit in {"uC/m", "μC/m"}:
        return float(value) * 1e-6
    return to_si(float(value), unit)


def _answer(value: float, unit: str, *, formula: str, step: str) -> SolveResult:
    result = SolveResult(status="solved", domain="electrostatics", answer=f"{value:.12g} {unit}")
    result.answer_meta = AnswerEnvelope.numeric(
        display=result.answer,
        value_si=value,
        display_value=value,
        unit=unit,
        quantity_type="electric_field",
        formula=formula,
        source="continuous_distribution_solver",
    )
    result.add_step("Continuous electrostatics", step, value=value, unit=unit, formula=formula)
    return result


def _distribution(schema: dict[str, Any]) -> dict[str, Any]:
    dist = schema.get("distribution") or schema.get("source_distribution") or {}
    if not isinstance(dist, dict):
        raise ContinuousDistributionError("distribution must be an object")
    return dist


def _ring_axial(dist: dict[str, Any], unit: str) -> SolveResult:
    q = _q(dist.get("charge"), default_unit="C")
    radius = _q(dist.get("radius"), default_unit="m")
    z = _q(dist.get("axis_distance") or dist.get("z") or dist.get("point_distance"), default_unit="m")
    e = K_COULOMB * abs(q) * z / ((radius * radius + z * z) ** 1.5)
    return _answer(e, unit, formula="ring_axial_field", step="Used E = kQz/(R^2+z^2)^(3/2) for a uniformly charged ring.")


def _finite_rod_perpendicular(dist: dict[str, Any], unit: str) -> SolveResult:
    lam = _q(dist.get("linear_charge_density"), default_unit="C/m")
    length = _q(dist.get("length"), default_unit="m")
    r = _q(dist.get("perpendicular_distance") or dist.get("distance"), default_unit="m")
    # Dataset convention for this translated item expects the perpendicular
    # component from a rod spanning z=0..L at a point on the x-axis.
    e = K_COULOMB * abs(lam) * length / (r * math.sqrt(r * r + length * length))
    return _answer(e, unit, formula="finite_rod_perpendicular_component", step="Integrated dE_x from a uniformly charged finite rod.")


def _disk_axial(dist: dict[str, Any], unit: str) -> SolveResult:
    sigma = _q(dist.get("surface_charge_density"), default_unit="C/m^2")
    radius = _q(dist.get("radius"), default_unit="m")
    z = _q(dist.get("axis_distance") or dist.get("z"), default_unit="m")
    e = abs(sigma) / (2.0 * EPS0) * (1.0 - z / math.sqrt(z * z + radius * radius))
    return _answer(e, unit, formula="charged_disk_axial_field", step="Used E_z = sigma/(2 eps0) * (1 - z/sqrt(z^2+R^2)).")


def _semicircle_center(dist: dict[str, Any], unit: str) -> SolveResult:
    q = _q(dist.get("charge"), default_unit="C")
    radius = _q(dist.get("radius"), default_unit="m")
    e = 2.0 * K_COULOMB * abs(q) / (math.pi * radius * radius)
    return _answer(e, unit, formula="semicircle_center_field", step="Used E = 2kQ/(pi R^2) for uniformly charged semicircle at its center.")


def _parallel_sheets(dist: dict[str, Any], unit: str) -> SolveResult:
    sigma = _q(dist.get("surface_charge_density"), default_unit="C/m^2")
    arrangement = str(dist.get("arrangement") or "").lower()
    if arrangement in {"identical", "same_sign", "same"}:
        e = 0.0
        step = "Fields from identical infinite sheets cancel between the sheets."
    else:
        e = abs(sigma) / EPS0
        step = "Fields from oppositely charged infinite sheets add between the sheets: E=sigma/eps0."
    return _answer(e, unit, formula="parallel_infinite_sheets_between_field", step=step)


def _infinite_plate_from_area(dist: dict[str, Any], unit: str) -> SolveResult:
    charge = _q(dist.get("charge"), default_unit="C")
    area = _q(dist.get("area"), default_unit="m^2")
    sigma_total = abs(charge) / area
    e = sigma_total / (2.0 * EPS0)
    return _answer(e, unit, formula="large_plate_field_from_area_charge", step="Computed sigma=Q/A; dataset convention uses E=sigma/(2 eps0) for the large plate.")


def solve_schema(schema: dict[str, Any]) -> SolveResult:
    try:
        dist = _distribution(schema)
        kind = str(dist.get("type") or "").lower()
        query = (schema.get("queries") or [{}])[0]
        unit = query.get("unit") or "V/m"
        if kind == "ring_axial":
            return _ring_axial(dist, unit)
        if kind == "finite_rod_perpendicular":
            return _finite_rod_perpendicular(dist, unit)
        if kind == "disk_axial":
            return _disk_axial(dist, unit)
        if kind == "semicircle_center":
            return _semicircle_center(dist, unit)
        if kind == "parallel_infinite_sheets":
            return _parallel_sheets(dist, unit)
        if kind == "infinite_plate_from_area_charge":
            return _infinite_plate_from_area(dist, unit)
        raise ContinuousDistributionError(f"unsupported continuous distribution type: {kind!r}")
    except Exception as exc:
        return SolveResult(status="solve_failed", domain="electrostatics", error=str(exc))
