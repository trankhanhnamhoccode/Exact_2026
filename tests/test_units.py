from xai_physics.core.units import to_si, convert


def test_charge_micro_coulomb_to_si():
    assert to_si(3, "μC") == 3e-6
    assert to_si(3, "uC") == 3e-6


def test_capacitance_pf_to_si():
    assert to_si(500, "pF") == 500e-12


def test_length_cm_to_m():
    assert to_si(10, "cm") == 0.1


def test_convert_pf_to_nf():
    assert convert(1000, "pF", "nF") == 1
