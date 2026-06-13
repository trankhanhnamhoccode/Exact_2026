from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Quantity:
    """A normalized quantity extracted from problem text."""

    value: str
    unit: str


@dataclass(frozen=True)
class FormulaSlot:
    """One non-query quantity slot needed by a formula candidate."""

    object_id: str
    quantity_key: str
    object_type: str | None = None
    required: bool = True
    default_value: str | None = None
    default_unit: str = ""
    role: str = "given"


@dataclass(frozen=True)
class FormulaSpec:
    """Declarative recipe for turning inventory + intent into a schema."""

    name: str
    query_intent: str
    query_id: str
    query_type: str
    query_unit: str
    slots: tuple[FormulaSlot, ...]


Inventory = Mapping[str, tuple[Quantity, ...]]


FORMULA_SPECS: tuple[FormulaSpec, ...] = (
    FormulaSpec(
        name="capacitor_charge_voltage",
        query_intent="charge",
        query_id="Q_query",
        query_type="charge",
        query_unit="nC",
        slots=(
            FormulaSlot("C1", "capacitance"),
            FormulaSlot("U1", "voltage"),
        ),
    ),
    FormulaSpec(
        name="point_charge_electric_field",
        query_intent="electric_field",
        query_id="E_query",
        query_type="electric_field",
        query_unit="V/m",
        slots=(
            FormulaSlot("q1", "charge"),
            FormulaSlot("r1", "distance"),
            FormulaSlot("eps_r", "relative_permittivity", required=False),
        ),
    ),
    FormulaSpec(
        name="electric_force_field",
        query_intent="electric_field",
        query_id="E_query",
        query_type="electric_field",
        query_unit="V/m",
        slots=(
            FormulaSlot("F1", "force"),
            FormulaSlot("q1", "charge"),
        ),
    ),
    FormulaSpec(
        name="electric_force_field",
        query_intent="charge",
        query_id="q_query",
        query_type="charge",
        query_unit="C",
        slots=(
            FormulaSlot("F1", "force"),
            FormulaSlot("E1", "electric_field"),
        ),
    ),
    FormulaSpec(
        name="electric_force_field",
        query_intent="force",
        query_id="F_query",
        query_type="force",
        query_unit="N",
        slots=(
            FormulaSlot("q1", "charge"),
            FormulaSlot("E1", "electric_field"),
        ),
    ),
)


def _schema(formula: str, objects: list[dict]) -> dict:
    return {
        "domain": "equations",
        "objects": objects,
        "relations": [{"type": "formula", "name": formula, "objects": [obj["id"] for obj in objects]}],
        "constraints": [],
    }


def _first(inventory: Inventory, key: str) -> Quantity | None:
    values = inventory.get(key) or ()
    return values[0] if values else None


def _object_from_quantity(slot: FormulaSlot, quantity: Quantity) -> dict:
    return {
        "id": slot.object_id,
        "type": slot.object_type or slot.quantity_key,
        "role": slot.role,
        "value": quantity.value,
        "unit": quantity.unit,
    }


def _object_from_default(slot: FormulaSlot) -> dict:
    return {
        "id": slot.object_id,
        "type": slot.object_type or slot.quantity_key,
        "role": "constant" if slot.role == "given" else slot.role,
        "value": slot.default_value,
        "unit": slot.default_unit,
    }


def fill_formula_specs(inventory: Inventory, query_intent: str | None) -> list[dict]:
    """Generate equation schema candidates from declarative FormulaSpec entries."""

    if query_intent is None:
        return []

    candidates: list[dict] = []
    for spec in FORMULA_SPECS:
        if spec.query_intent != query_intent:
            continue

        objects: list[dict] = []
        missing_required = False
        for slot in spec.slots:
            quantity = _first(inventory, slot.quantity_key)
            if quantity is not None:
                objects.append(_object_from_quantity(slot, quantity))
                continue
            if slot.default_value is not None:
                objects.append(_object_from_default(slot))
                continue
            if slot.required:
                missing_required = True
                break

        if missing_required:
            continue

        objects.append(
            {
                "id": spec.query_id,
                "type": spec.query_type,
                "role": "query",
                "value": None,
                "unit": spec.query_unit,
            }
        )
        candidates.append(_schema(spec.name, objects))

    return candidates
