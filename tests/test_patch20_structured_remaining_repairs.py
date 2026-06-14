from xai_physics.domains.capacitor_state.engine import solve_schema as solve_capacitor_state
from xai_physics.domains.electrostatics.coordinate_builder import build_coordinates
from xai_physics.domains.electrostatics.engine import solve_schema as solve_electrostatics
from xai_physics.domains.equations.solver import solve_schema as solve_equations
from xai_physics.eval.replay_llm_dataset import compare_answer_with_meta


def test_collinear_path_repair_overwrites_bad_llm_order_dt092_shape():
    schema = {
        "domain": "electrostatics",
        "points": [{"id": p} for p in ["A", "B", "C", "M", "N"]],
        "geometry": [{
            "type": "Collinear",
            "points": ["A", "B", "C", "M", "N"],
            "order": ["A", "B", "C", "M", "N"],
            "distances": [
                {"between": ["A", "B"], "value": 10, "unit": "cm"},
                {"between": ["B", "C"], "value": 10, "unit": "cm"},
                {"between": ["A", "M"], "value": 10, "unit": "cm"},
                {"between": ["C", "N"], "value": 10, "unit": "cm"},
            ],
        }],
        "charges": [
            {"id": "q1", "charge": {"value": -2e-6, "unit": "C"}, "at": "A"},
            {"id": "q2", "charge": {"value": 3e-6, "unit": "C"}, "at": "B"},
            {"id": "q3", "charge": {"value": -1e-6, "unit": "C"}, "at": "C"},
        ],
        "queries": [{"type": "electric_field", "target": "M", "output": "magnitude", "unit": "V/m"}],
    }

    coords = build_coordinates(schema)
    assert coords["M"].x < coords["A"].x < coords["B"].x < coords["C"].x < coords["N"].x

    result = solve_electrostatics(schema)
    assert result.status == "solved"
    assert compare_answer_with_meta(result.answer, None, "1.23 . 10^6", expected_unit="V/m") is True


def test_collinear_order_can_use_pairwise_distances_from_separate_geometry_dt093_shape():
    schema = {
        "domain": "electrostatics",
        "points": [{"id": p} for p in ["A", "B", "C", "M", "N"]],
        "geometry": [
            {"type": "Collinear", "points": ["A", "B", "C"], "order": ["A", "B", "C"]},
            {"type": "PairwiseDistances", "points": ["A", "B", "C", "M", "N"], "distances": [
                {"between": ["A", "B"], "value": 10, "unit": "cm"},
                {"between": ["B", "C"], "value": 10, "unit": "cm"},
                {"between": ["M", "A"], "value": 10, "unit": "cm"},
                {"between": ["N", "C"], "value": 10, "unit": "cm"},
            ]},
        ],
        "charges": [
            {"id": "q1", "charge": {"value": -2, "unit": "μC"}, "at": "A"},
            {"id": "q2", "charge": {"value": 3, "unit": "μC"}, "at": "B"},
            {"id": "q3", "charge": {"value": -1, "unit": "μC"}, "at": "C"},
        ],
        "queries": [{"type": "electric_field", "target": "N", "output": "magnitude", "unit": "V/m"}],
    }

    result = solve_electrostatics(schema)
    assert result.status == "solved"
    assert compare_answer_with_meta(result.answer, None, "4.25 . 10^5", expected_unit="V/m") is True


def test_capacitor_state_keeps_numeric_answer_but_exposes_percent_reduction():
    schema = {
        "domain": "capacitor_state",
        "entities": [{
            "id": "C1", "type": "capacitor",
            "capacitance": {"value": 8, "unit": "uF"},
            "voltage": {"value": 10, "unit": "V"},
            "connected_to_source": True,
        }],
        "events": [{"type": "ReplaceCapacitor", "apply_to": ["C1"], "params": {"new_capacitance": {"value": 4, "unit": "uF"}, "hold": "voltage"}}],
        "queries": [{"type": "energy_reduction", "target": "C1", "unit": "uJ"}],
    }

    result = solve_capacitor_state(schema)
    assert result.status == "solved"
    assert result.answer == "200 uJ"
    assert result.answer_meta is not None
    assert compare_answer_with_meta(result.answer, result.answer_meta.as_dict(), "50%") is True


def test_capacitor_state_qualitative_unchanged_and_halved_use_answer_meta():
    unchanged = {
        "domain": "capacitor_state",
        "entities": [{"id": "C1", "type": "capacitor", "capacitance": {"value": 1, "unit": "uF"}, "voltage": {"value": 100, "unit": "V"}, "connected_to_source": False}],
        "events": [{"type": "InsertDielectric", "apply_to": ["C1"], "params": {"dielectric_constant": 2}}],
        "queries": [{"type": "charge", "target": "C1", "unit": "uC"}],
    }
    unchanged_result = solve_capacitor_state(unchanged)
    assert unchanged_result.answer == "100 uC"
    assert compare_answer_with_meta(unchanged_result.answer, unchanged_result.answer_meta.as_dict(), "Do not change") is True

    halved = {
        "domain": "capacitor_state",
        "entities": [{"id": "C1", "type": "capacitor", "capacitance": {"value": 1, "unit": "uF"}, "voltage": {"value": 10, "unit": "V"}, "connected_to_source": False}],
        "events": [{"type": "InsertDielectric", "apply_to": ["C1"], "params": {"dielectric_constant": 2}}],
        "queries": [{"type": "voltage", "target": "C1", "unit": "V"}],
    }
    halved_result = solve_capacitor_state(halved)
    assert halved_result.answer == "5 V"
    assert compare_answer_with_meta(halved_result.answer, halved_result.answer_meta.as_dict(), "the voltage is halfed") is True


def test_replace_dielectric_capacitance_ratio_from_k_without_absolute_capacitance():
    schema = {
        "domain": "capacitor_state",
        "entities": [{"id": "C1", "type": "capacitor", "capacitance": {}, "connected_to_source": False}],
        "events": [{"type": "ReplaceDielectric", "apply_to": ["C1"], "params": {"initial_k": 4, "final_k": 2}}],
        "queries": [{"type": "capacitance_ratio", "target": "C1", "unit": "times"}],
    }

    result = solve_capacitor_state(schema)
    assert result.status == "solved"
    assert result.answer == "0.5 times"
    assert compare_answer_with_meta(result.answer, None, "decreases by half") is True


def test_energy_sequence_same_capacitor_returns_initial_final_and_ratio():
    schema = {
        "domain": "equations",
        "objects": [
            {"id": "Q1", "type": "charge", "role": "given", "value": "90", "unit": "uC"},
            {"id": "Q2", "type": "charge", "role": "given", "value": "45", "unit": "uC"},
            {"id": "U1", "type": "voltage", "role": "given", "value": "30", "unit": "V"},
            {"id": "W1", "type": "energy", "role": "query", "unit": "uJ"},
            {"id": "W2", "type": "energy", "role": "query", "unit": "uJ"},
        ],
        "relations": [{"type": "formula", "name": "capacitor_energy_charge_voltage", "objects": ["Q1", "U1", "W1", "Q2", "W2"]}],
    }

    result = solve_equations(schema)
    assert result.status == "ok"
    assert "initial=1350 uJ" in result.answer
    assert "final=337.5 uJ" in result.answer
    assert compare_answer_with_meta(result.answer, None, "decreases by 4 times") is True
