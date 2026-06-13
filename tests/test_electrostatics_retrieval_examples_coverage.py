from xai_physics.domains.electrostatics.contract import SUPPORTED_GEOMETRY, validate_schema
from xai_physics.domains.electrostatics.retrieval.example_store import load_examples


def test_electrostatics_examples_cover_each_geometry_primitive():
    examples = load_examples()
    covered = set()
    for ex in examples:
        for geom in ex.schema.get("geometry", []):
            gtype = geom.get("type")
            if gtype:
                covered.add(gtype)

    missing = SUPPORTED_GEOMETRY - covered
    assert not missing, f"missing examples for geometry primitives: {sorted(missing)}"


def test_electrostatics_examples_include_direct_vector_and_coordinate_cases():
    examples = load_examples()
    assert any(ex.schema.get("vectors") for ex in examples)
    assert any(
        any("x" in p and "y" in p for p in ex.schema.get("points", []))
        for ex in examples
    )


def test_electrostatics_examples_validate():
    for ex in load_examples():
        validate_schema(ex.schema)
