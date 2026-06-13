from __future__ import annotations

from typing import Any, Optional

from xai_physics.core.result import SolveResult
from xai_physics.core.units import to_si, convert
from xai_physics.domains.capacitor_state.state import CapacitorState
from xai_physics.domains.capacitor_state.system import SystemState
from xai_physics.domains.capacitor_state.events import (
    AreaScale,
    ConnectToSource,
    DisconnectFromSource,
    DistanceScale,
    InsertDielectric,
    ShortCircuit,
    ConnectToInductor,
    ReplaceDielectric,
    ReplaceCapacitor,

)
from xai_physics.domains.capacitor_state.redistribution import ParallelRedistribution
from xai_physics.domains.capacitor_state.contract import validate_schema


def _quantity_to_si(data: Optional[dict[str, Any]]) -> Optional[float]:
    if data is None:
        return None

    value = data.get("value")
    unit = data.get("unit")

    if value is None or unit is None:
        return None

    return to_si(float(value), str(unit))


def _build_system(schema: dict[str, Any]) -> SystemState:
    system = SystemState()

    for ent in schema.get("entities", []):
        if ent.get("type", "capacitor") != "capacitor":
            continue

        cap = CapacitorState(
            id=ent["id"],
            capacitance_F=_quantity_to_si(ent.get("capacitance")),
            voltage_V=_quantity_to_si(ent.get("voltage")),
            charge_C=_quantity_to_si(ent.get("charge")),
            connected_to_source=bool(ent.get("connected_to_source", False)),
        )
        cap.infer_missing()
        system.add(cap)

    system.infer_all()
    return system


def _apply_single_cap_event(event_schema: dict[str, Any], system: SystemState) -> str:
    event_type = event_schema["type"]
    apply_to = event_schema.get("apply_to") or event_schema.get("target")
    params = event_schema.get("params", {})

    if isinstance(apply_to, list):
        if len(apply_to) != 1:
            raise ValueError(f"Event {event_type} expects exactly one capacitor.")
        cap_id = apply_to[0]
    else:
        cap_id = apply_to

    if cap_id is None:
        raise ValueError(f"Missing apply_to for event {event_type}.")

    cap = system.get(cap_id)

    if event_type == "ConnectToSource":
        voltage = _quantity_to_si(params.get("voltage"))
        if voltage is None:
            raise ValueError("ConnectToSource requires params.voltage.")
        return ConnectToSource(voltage_V=voltage).apply(cap)

    if event_type == "DisconnectFromSource":
        return DisconnectFromSource().apply(cap)

    if event_type == "ReplaceDielectric":
        initial_k = params.get("initial_k")
        final_k = params.get("final_k")
        if initial_k is None or final_k is None:
            raise ValueError("ReplaceDielectric requires params.initial_k and params.final_k.")
        return ReplaceDielectric(initial_k=initial_k, final_k=final_k).apply(cap)

    if event_type in {"ReplaceCapacitor", "SetCapacitance"}:
        new_cap_data = (
            params.get("new_capacitance")
            or params.get("capacitance")
            or params.get("final_capacitance")
        )
        new_cap = _quantity_to_si(new_cap_data)
        if new_cap is None:
            raise ValueError("ReplaceCapacitor requires params.new_capacitance.")

        hold = (
            params.get("hold")
            or params.get("hold_policy")
            or params.get("voltage_policy")
            or "auto"
        )
        return ReplaceCapacitor(new_capacitance_F=new_cap, hold_policy=str(hold)).apply(cap)

    if event_type == "InsertDielectric":
        k = params.get("dielectric_constant", params.get("k"))
        if k is None:
            raise ValueError("InsertDielectric requires dielectric_constant or k.")
        return InsertDielectric(dielectric_constant=float(k)).apply(cap)

    if event_type == "DistanceScale":
        factor = params.get("factor")
        if factor is None:
            raise ValueError("DistanceScale requires factor.")
        return DistanceScale(factor=float(factor)).apply(cap)

    if event_type == "AreaScale":
        factor = params.get("factor")
        if factor is None:
            raise ValueError("AreaScale requires factor.")
        return AreaScale(factor=float(factor)).apply(cap)

    if event_type == "ShortCircuit":
        return ShortCircuit().apply(cap)

    if event_type == "ConnectToInductor":
        ConnectToInductor().apply(cap)
        return

    raise ValueError(f"Unsupported single-capacitor event: {event_type}")


def _apply_system_event(event_schema: dict[str, Any], system: SystemState) -> str:
    event_type = event_schema["type"]

    if event_type == "ParallelRedistribution":
        cap_ids = event_schema.get("apply_to", [])
        params = event_schema.get("params", {})
        polarity = params.get("polarity", "same")

        if polarity != "same":
            raise ValueError(
                "ParallelRedistribution currently supports only same-polarity connection."
            )

        return ParallelRedistribution(capacitor_ids=cap_ids).apply(system)

    return _apply_single_cap_event(event_schema, system)


def _format_value(value_si: float, si_unit: str, output_unit: Optional[str]) -> str:
    if output_unit is None:
        output_unit = si_unit

    value = convert(value_si, si_unit, output_unit)
    return f"{value:g} {output_unit}"


def _answer_query(query: dict[str, Any], system: SystemState, initial_system: SystemState | None = None) -> str:
    if query.get("type") == "energy_percent":
        if initial_system is None:
            raise ValueError("energy_percent query requires initial_system snapshot.")

        target = query.get("target", "system")

        if target == "system":
            initial_energy = sum(
                cap.energy_J()
                for cap in initial_system.capacitors.values()
            )
            final_energy = sum(
                cap.energy_J()
                for cap in system.capacitors.values()
            )
        else:
            initial_energy = initial_system.get(target).energy_J()
            final_energy = system.get(target).energy_J()

        if initial_energy == 0:
            raise ValueError("Cannot compute energy_percent because initial energy is zero.")

        percent = 100.0 * final_energy / initial_energy
        return f"{percent:g} %"

    if query.get("type") == "capacitance_ratio":
        if initial_system is None:
            raise ValueError("capacitance_ratio query requires initial_system snapshot.")

        target = query.get("target", "system")

        if target == "system":
            initial_c = sum(cap.capacitance_F for cap in initial_system.capacitors.values())
            final_c = sum(cap.capacitance_F for cap in system.capacitors.values())
        else:
            initial_c = initial_system.get(target).capacitance_F
            final_c = system.get(target).capacitance_F

        if initial_c == 0:
            raise ValueError("Cannot compute capacitance_ratio because initial capacitance is zero.")

        ratio = final_c / initial_c
        return f"{ratio:g} times"

    if query.get("type") == "energy_ratio":
        if initial_system is None:
            raise ValueError("energy_ratio query requires initial_system snapshot.")

        target = query.get("target", "system")

        if target == "system":
            initial_energy = sum(
                cap.energy_J()
                for cap in initial_system.capacitors.values()
            )
            final_energy = sum(
                cap.energy_J()
                for cap in system.capacitors.values()
            )
        else:
            initial_energy = initial_system.get(target).energy_J()
            final_energy = system.get(target).energy_J()

        if initial_energy == 0:
            raise ValueError("Cannot compute energy_ratio because initial energy is zero.")

        ratio = final_energy / initial_energy
        return f"{ratio:g} times"

    system.infer_all()

    qtype = query.get("type", "voltage")
    target = query.get("target", "system")
    output_unit = query.get("unit")

    if qtype in {"energy_change", "energy_reduction"}:
        if initial_system is None:
            raise ValueError(f"{qtype} query requires initial_system snapshot.")
        initial_energy = _energy_of_target(initial_system, target)
        final_energy = _energy_of_target(system, target)
        value_si = final_energy - initial_energy
        if qtype == "energy_reduction":
            value_si = initial_energy - final_energy
        return _format_value(value_si, "J", output_unit)

    if target == "system":

        if qtype == "energy":

            value_si = sum(

                cap.energy_J()

                for cap in system.capacitors.values()

            )

            return _format_value(value_si, "J", output_unit)

        caps = list(system.capacitors.values())
        if not caps:
            raise ValueError("System has no capacitors.")

        if qtype == "voltage":
            voltage = caps[0].voltage_V
            if voltage is None:
                raise ValueError("System voltage is unknown.")
            return _format_value(voltage, "V", output_unit)

        if qtype == "charge":
            total = 0.0
            for cap in caps:
                cap.infer_missing()
                if cap.charge_C is None:
                    raise ValueError(f"Charge is unknown for capacitor {cap.id}.")
                total += cap.charge_C
            return _format_value(total, "C", output_unit)

        if qtype == "capacitance":
            total = 0.0
            for cap in caps:
                if cap.capacitance_F is None:
                    raise ValueError(f"Capacitance is unknown for capacitor {cap.id}.")
                total += cap.capacitance_F
            return _format_value(total, "F", output_unit)

        if qtype == "energy":
            total = 0.0
            for cap in caps:
                energy = cap.energy_J()
                if energy is None:
                    raise ValueError(f"Energy is unknown for capacitor {cap.id}.")
                total += energy
            return _format_value(total, "J", output_unit)

    cap = system.get(target)
    cap.infer_missing()

    if qtype == "voltage":
        if cap.voltage_V is None:
            raise ValueError(f"Voltage is unknown for capacitor {target}.")
        return _format_value(cap.voltage_V, "V", output_unit)

    if qtype == "charge":
        if cap.charge_C is None:
            raise ValueError(f"Charge is unknown for capacitor {target}.")
        return _format_value(cap.charge_C, "C", output_unit)

    if qtype == "capacitance":
        if cap.capacitance_F is None:
            raise ValueError(f"Capacitance is unknown for capacitor {target}.")
        return _format_value(cap.capacitance_F, "F", output_unit)

    if qtype == "energy":
        energy = cap.energy_J()
        if energy is None:
            raise ValueError(f"Energy is unknown for capacitor {target}.")
        return _format_value(energy, "J", output_unit)

    raise ValueError(f"Unsupported query type: {qtype}")


def _clone_quantity(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    return dict(data)


def _repair_replace_capacitor_schema(schema: dict[str, Any]) -> None:
    """
    Repair a common LLM confusion:
    "replaced by another capacitor with capacitance ..." is not ReplaceDielectric.

    The bad schema often contains two capacitor entities plus a ReplaceDielectric event
    with empty params, sometimes followed by ParallelRedistribution. Convert it into
    ReplaceCapacitor on the first capacitor using the second capacitor's capacitance.
    """
    events = schema.get("events", [])
    entities = schema.get("entities", [])

    if not isinstance(events, list) or not isinstance(entities, list) or len(entities) < 2:
        return

    bad_index = None
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if event.get("type") != "ReplaceDielectric":
            continue
        params = event.get("params") or {}
        if params.get("initial_k") is None and params.get("final_k") is None:
            bad_index = i
            break

    if bad_index is None:
        return

    first = entities[0]
    second = entities[1]
    new_cap = _clone_quantity(second.get("capacitance"))
    if not new_cap or new_cap.get("value") is None:
        return

    hold = "voltage"
    v1 = _clone_quantity(first.get("voltage"))
    v2 = _clone_quantity(second.get("voltage"))
    if v1 and v2 and v1.get("value") == v2.get("value"):
        hold = "voltage"
    elif not first.get("connected_to_source", False):
        hold = "charge"

    events[bad_index] = {
        "type": "ReplaceCapacitor",
        "apply_to": [first.get("id", "C1")],
        "params": {
            "new_capacitance": new_cap,
            "hold": hold,
        },
    }

    # Drop a spurious redistribution event created from "replaced by another capacitor".
    schema["events"] = [
        event for event in events
        if not (isinstance(event, dict) and event.get("type") == "ParallelRedistribution")
    ]

    # The bad schema frequently asks energy_ratio for wording "reduction in energy".
    # Use an absolute energy reduction query; this is safer for replacement wording.
    queries = schema.get("queries", [])
    if isinstance(queries, list):
        for query in queries:
            if isinstance(query, dict) and query.get("type") == "energy_ratio":
                query["type"] = "energy_reduction"
                if query.get("unit") in {None, "", "times", "x"}:
                    query["unit"] = "uJ"


def _energy_of_target(system: SystemState, target: str) -> float:
    system.infer_all()

    if target == "system":
        total = 0.0
        for cap in system.capacitors.values():
            energy = cap.energy_J()
            if energy is None:
                raise ValueError(f"Energy is unknown for capacitor {cap.id}.")
            total += energy
        return total

    energy = system.get(target).energy_J()
    if energy is None:
        raise ValueError(f"Energy is unknown for capacitor {target}.")
    return energy


def solve_schema(schema: dict[str, Any]) -> SolveResult:
    result = SolveResult(status="solved", domain="capacitor_state")

    try:
        _repair_replace_capacitor_schema(schema)
        validate_schema(schema)
        system = _build_system(schema)

        initial_system = system.clone()
        result.add_step(
            "Build system",
            "Built capacitor system from canonical schema.",
            capacitors={
                cid: {
                    "capacitance_F": cap.capacitance_F,
                    "voltage_V": cap.voltage_V,
                    "charge_C": cap.charge_C,
                    "connected_to_source": cap.connected_to_source,
                }
                for cid, cap in system.capacitors.items()
            },
        )

        for event_schema in schema.get("events", []):
            detail = _apply_system_event(event_schema, system)
            system.infer_all()

            result.add_step(
                f"Apply event: {event_schema['type']}",
                detail,
                capacitors={
                    cid: {
                        "capacitance_F": cap.capacitance_F,
                        "voltage_V": cap.voltage_V,
                        "charge_C": cap.charge_C,
                        "connected_to_source": cap.connected_to_source,
                    }
                    for cid, cap in system.capacitors.items()
                },
            )

        queries = schema.get("queries", [])
        if not queries:
            raise ValueError("Schema has no query.")

        answers = [
            _answer_query(query, system, initial_system=initial_system)
            for query in queries
        ]
        answer = "; ".join(answers)
        result.answer = answer
        result.add_step(
            "Final answer",
            f"The requested output is {answer}.",
        )

        return result

    except Exception as exc:
        result.status = "solve_failed"
        result.error = str(exc)
        return result



