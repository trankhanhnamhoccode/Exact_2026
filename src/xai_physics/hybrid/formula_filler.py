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
    trigger_any: tuple[str, ...] = ()
    trigger_all: tuple[str, ...] = ()


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
        name="capacitor_charge_voltage",
        query_intent="capacitance",
        query_id="C_query",
        query_type="capacitance",
        query_unit="uF",
        slots=(
            FormulaSlot("Q1", "charge"),
            FormulaSlot("U1", "voltage"),
        ),
    ),
    # Put F = |q|E before point-charge field: force data is usually more
    # direct than inferring E from the source charge when both are present.
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
        trigger_any=("point charge", "small sphere", "electric charge", "charge q", "charge of"),
    ),
    FormulaSpec(
        name="infinite_wire_electric_field",
        query_intent="electric_field",
        query_id="E_query",
        query_type="electric_field",
        query_unit="V/m",
        slots=(
            FormulaSlot("lambda1", "line_charge_density"),
            FormulaSlot("r1", "distance"),
        ),
        trigger_any=("wire", "linear charge density", "λ", "lambda"),
    ),
    FormulaSpec(
        name="equilibrium_electric_field",
        query_intent="electric_field",
        query_id="E_query",
        query_type="electric_field",
        query_unit="V/m",
        slots=(
            FormulaSlot("m1", "mass"),
            FormulaSlot("q1", "charge"),
            FormulaSlot("g1", "gravity", object_type="gravitational_acceleration", required=False, default_value="10", default_unit="m/s2"),
        ),
        trigger_all=("equilibrium",),
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


def _matches_text(spec: FormulaSpec, text: str) -> bool:
    low = text.lower()
    if spec.trigger_all and not all(term.lower() in low for term in spec.trigger_all):
        return False
    if spec.trigger_any and not any(term.lower() in low for term in spec.trigger_any):
        return False
    return True


def fill_formula_specs(inventory: Inventory, query_intent: str | None, *, text: str = "") -> list[dict]:
    """Generate equation schema candidates from declarative FormulaSpec entries."""

    if query_intent is None:
        return []

    candidates: list[dict] = []
    for spec in FORMULA_SPECS:
        if spec.query_intent != query_intent or not _matches_text(spec, text):
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
