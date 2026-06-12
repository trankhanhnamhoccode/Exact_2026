import pytest

from xai_physics.core.units import to_si, convert


def test_micro_faraday_aliases():
    expected = pytest.approx(25e-6)

    assert to_si(25, "uF") == expected
    assert to_si(25, "μF") == expected
    assert to_si(25, "µF") == expected
    assert to_si(25, "?F") == expected


def test_micro_coulomb_aliases():
    expected = pytest.approx(30e-6)

    assert to_si(30, "uC") == expected
    assert to_si(30, "μC") == expected
    assert to_si(30, "µC") == expected
    assert to_si(30, "?C") == expected


def test_energy_conversion_to_mj():
    assert convert(0.18, "J", "mJ") == pytest.approx(180)
