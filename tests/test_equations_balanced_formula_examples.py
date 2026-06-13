from xai_physics.domains.equations.retrieval.example_store import load_examples
from xai_physics.domains.equations.retrieval.pipeline import retrieve_equations_context
from xai_physics.llm.prompt_builder import build_schema_prompt


def _query_types_by_formula():
    out = {}
    for ex in load_examples():
        qtypes = {
            obj.get("type")
            for obj in ex.schema.get("objects", [])
            if obj.get("role") == "query"
        }
        out.setdefault(ex.formula_id, set()).update(qtypes)
    return out


def test_capacitor_core_formulas_have_balanced_query_examples():
    qtypes = _query_types_by_formula()

    assert {"charge", "capacitance", "voltage"} <= qtypes["capacitor_charge_voltage"]
    assert {"energy", "capacitance", "voltage"} <= qtypes["capacitor_energy_voltage"]
    assert {"energy", "charge", "voltage"} <= qtypes["capacitor_energy_charge_voltage"]


def test_circuit_lc_and_field_formulas_have_non_default_query_examples():
    qtypes = _query_types_by_formula()

    assert {"electric_field", "voltage", "distance"} <= qtypes["parallel_plate_field"]
    assert {"frequency", "capacitance", "inductance"} <= qtypes["lc_resonance_frequency"]
    assert {"current", "voltage", "resistance"} <= qtypes["ohm_law"]
    assert {"power", "voltage", "resistance"} <= qtypes["power_voltage_resistance"]
    assert {"energy", "current", "inductance"} <= qtypes["inductor_energy"]


def test_voltage_scaling_retrieval_uses_ratio_example_and_formula():
    q = (
        "A capacitor has a capacitance of 2 μF, and a voltage U is applied across it. "
        "If U is doubled, how many times will the electric field energy increase?"
    )
    ctx = retrieve_equations_context(q, formula_top_k=5, example_top_k=5)

    assert ctx.selected_formulas[0].formula.id == "capacitor_energy_voltage_scaling_constant_capacitance"
    assert "eq_ex_cap_energy_voltage_scaling_doubled_td367" in [
        item.example.id for item in ctx.selected_examples
    ]


def test_equations_prompt_includes_more_than_two_examples_when_k_is_two():
    q = "A capacitor stores 160 μJ of energy at a voltage of 8 V. Calculate the charge on the capacitor."
    built = build_schema_prompt(q, k=2)

    assert built.domain_decision.domain == "equations"
    assert len(built.examples) >= 5
    assert "Formula schema_template is only a structural guide" in built.prompt
