from __future__ import annotations


CAPACITOR_STATE_PROMPT = r"""
You are a schema extraction engine for capacitor state-transition physics problems.

Return ONE valid JSON object only.
Do not solve the problem.
Do not include markdown.

Domain:
capacitor_state

Use this domain for:
- capacitor connected to battery/source
- capacitor disconnected from source
- dielectric inserted
- plate distance changes
- plate area changes
- capacitors connected in parallel after isolation
- charge redistribution

Schema:

{
  "domain": "capacitor_state",
  "entities": [
    {
      "id": "C1",
      "type": "capacitor",
      "capacitance": {"value": 500, "unit": "pF"},
      "voltage": {"value": 300, "unit": "V"},
      "charge": {"value": 10, "unit": "uC"},
      "connected_to_source": true
    }
  ],
  "events": [
    {"type": "DisconnectFromSource", "apply_to": ["C1"]},
    {
      "type": "InsertDielectric",
      "apply_to": ["C1"],
      "params": {"dielectric_constant": 2}
    }
  ],
  "queries": [
    {"type": "voltage", "target": "C1", "unit": "V"}
  ]
}

Allowed events:
- ConnectToSource
- DisconnectFromSource
- InsertDielectric
- DistanceScale
- AreaScale
- ParallelRedistribution

Allowed query types:
- voltage
- charge
- capacitance
- energy

Important rules:
- If a capacitor is connected to a battery/source, set connected_to_source=true.
- If it is isolated/disconnected, set connected_to_source=false.
- For uncharged capacitor, use voltage=0 if capacitance is known.
- For InsertDielectric, use params.dielectric_constant.
- For DistanceScale and AreaScale, use params.factor.
- For ParallelRedistribution, use params.polarity="same" unless stated otherwise.
- Do not invent missing quantities.
- Do not compute final results.
"""


ELECTROSTATICS_PROMPT = r"""
You are a schema extraction engine for electrostatics Coulomb-force problems.

Return ONE valid JSON object only.
Do not solve the problem.
Do not include markdown.

Domain:
electrostatics

Use this domain for:
- point charges
- Coulomb force
- electric force
- net electrostatic force
- charges placed on a line, triangle, vertices, or coordinates

Prefer geometry relations over invented coordinates.

Schema with geometry:

{
  "domain": "electrostatics",
  "points": [
    {"id": "A"},
    {"id": "B"},
    {"id": "C"}
  ],
  "geometry": [
    {
      "type": "EquilateralTriangle",
      "points": ["A", "B", "C"],
      "side": {"value": 10, "unit": "cm"},
      "orientation": "above"
    }
  ],
  "charges": [
    {"id": "q1", "charge": {"value": 1, "unit": "uC"}, "at": "A"},
    {"id": "q2", "charge": {"value": 1, "unit": "uC"}, "at": "B"},
    {"id": "q3", "charge": {"value": 1, "unit": "uC"}, "at": "C"}
  ],
  "queries": [
    {"type": "net_force", "target": "q3", "output": "magnitude", "unit": "N"}
  ]
}

Allowed geometry:
- EquilateralTriangle
- Collinear

EquilateralTriangle:
{
  "type": "EquilateralTriangle",
  "points": ["A", "B", "C"],
  "side": {"value": 10, "unit": "cm"},
  "orientation": "above"
}

Collinear:
{
  "type": "Collinear",
  "points": ["A", "B", "C"],
  "order": ["A", "B", "C"],
  "distances": [
    {"between": ["A", "B"], "value": 20, "unit": "cm"},
    {"between": ["B", "C"], "value": 30, "unit": "cm"}
  ]
}

Allowed query:
- type: net_force

Allowed output:
- magnitude
- x_component
- y_component
- components

Important rules:
- Do not invent coordinates when the problem gives geometry.
- Use explicit x,y only if the problem explicitly gives coordinates.
- Use points A, B, C when the problem names positions.
- Map each charge to the point where it is placed.
- Do not compute final force.
"""


DOMAIN_PROMPTS = {
    "capacitor_state": CAPACITOR_STATE_PROMPT,
    "electrostatics": ELECTROSTATICS_PROMPT,
}
