import numpy as np
import pytest

from Library.dimension_lib import (
    create_2d_coordinate_grid,
    is_cylindrical_order,
    map_k_by_order,
)


def test_cartesian_order_is_still_a_direct_permutation():
    assert map_k_by_order(1.0, 2.0, 3.0, "yzx") == (3.0, 1.0, 2.0)
    assert not is_cylindrical_order("yzx")


@pytest.mark.parametrize(
    ("order", "at_zero", "at_quarter_turn"),
    [
        ("xpz", (2.0, 0.0, 5.0), (0.0, 2.0, 5.0)),
        ("ypz", (0.0, 2.0, 5.0), (-2.0, 0.0, 5.0)),
        ("xpy", (2.0, 5.0, 0.0), (0.0, 5.0, -2.0)),
        ("zpy", (0.0, 5.0, 2.0), (2.0, 5.0, 0.0)),
        ("ypx", (5.0, 2.0, 0.0), (5.0, 0.0, 2.0)),
        ("zpx", (5.0, 0.0, 2.0), (5.0, -2.0, 0.0)),
    ],
)
def test_cylindrical_orders_use_right_handed_phi(order, at_zero, at_quarter_turn):
    assert is_cylindrical_order(order)
    assert np.allclose(map_k_by_order(2.0, 0.0, 5.0, order), at_zero)
    assert np.allclose(
        map_k_by_order(2.0, np.pi / 2.0, 5.0, order),
        at_quarter_turn,
        atol=1e-14,
    )


def test_cylindrical_grid_uses_nonnegative_radius_without_duplicate_phi_endpoint():
    radius, phi, info = create_2d_coordinate_grid(
        3.0,
        8,
        order="xpz",
        include_endpoints=True,
    )

    assert radius.shape == (8, 8)
    assert phi.shape == (8, 8)
    assert np.min(radius) == 0.0
    assert np.max(radius) == 3.0
    assert np.min(phi) == 0.0
    assert np.max(phi) < 2.0 * np.pi
    assert info["phi_domain"] == [0.0, 2.0 * np.pi]
    assert info["phi_endpoint_included"] is False


def test_cylindrical_grid_maps_vectorized_to_cartesian_arrays():
    radius, phi, _ = create_2d_coordinate_grid(2.0, 4, order="xpz")
    kx, ky, kz = map_k_by_order(radius, phi, 0.25, "xpz")

    assert kx.shape == radius.shape
    assert ky.shape == radius.shape
    assert kz.shape == radius.shape
    assert np.allclose(np.sqrt(kx**2 + ky**2), radius)
    assert np.allclose(kz, 0.25)


@pytest.mark.parametrize("order", ["pxz", "xyzp", "xpp", "abc"])
def test_invalid_coordinate_orders_are_rejected(order):
    with pytest.raises(ValueError):
        map_k_by_order(1.0, 2.0, 0.0, order)
