from xai_physics.router.domain_router import route_domain


def test_route_equation():
    q = "Calculate the current when U = 80 V and R = 40 ohm."
    assert route_domain(q) == "equations"


def test_route_electrostatics_geometry():
    q = "Three charges are placed at the vertices of an equilateral triangle. Find the net electrostatic force on q3."
    assert route_domain(q) == "electrostatics"


def test_route_capacitor_state():
    q = "A capacitor is initially connected to a battery, then disconnected and a dielectric is inserted. Find the final voltage."
    assert route_domain(q) == "capacitor_state"
