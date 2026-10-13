"""Physical regression checks for the exact SG194 nodal-network tracer.

Run next to the plotting script. For a staging directory outside QGT_Lib, set
QGT_LIB_ROOT to the repository containing Library/Hamiltonian.
"""

import numpy as np
import pytest

import Plot_gWave_Nodal_Network_3D as network


@pytest.fixture(scope="module")
def default_network():
    return network.trace_network(grid=151)


def _points(curves):
    assert curves, "Expected a nonempty nodal network"
    return np.concatenate([curve.xyz for curve in curves])


def _direct_middle_gap(points, params):
    """Check the real four-band eigensystem, independently of tracer equations."""
    hamiltonian = network.gWaveAltermagnetHamiltonian(**params)
    points = np.asarray(points, dtype=float)
    matrices = hamiltonian.compute_static_vectorized(*points.T)
    energies = np.linalg.eigvalsh(matrices)
    return energies[:, 2] - energies[:, 1]


def _assert_near(points, target, tolerance):
    distance = np.linalg.norm(points - np.asarray(target), axis=1)
    assert distance.min() < tolerance, (
        f"Missing branch near {target}; closest vertex is {distance.min():.6g} away"
    )


def test_default_network_is_degenerate_and_retains_both_height_branches(default_network):
    curves, junctions = default_network
    points = _points(curves)
    assert np.isfinite(points).all()
    assert network.in_bz(points[:, :2]).all()
    assert np.max(np.abs(points[:, 2])) <= np.pi + 1e-10
    assert np.max(_direct_middle_gap(points, network.PARAMETERS)) < 1e-8
    assert np.max(_direct_middle_gap(junctions, network.PARAMETERS)) < 1e-8

    # At kx=kz=0, 3 sin²(sqrt(3) ky/2)=Jz/lamb=2.
    # The two positive roots and their negatives belong to the first BZ.
    inner_y = 2 / np.sqrt(3) * np.arcsin(np.sqrt(2 / 3))
    outer_y = 2 / np.sqrt(3) * (np.pi - np.arcsin(np.sqrt(2 / 3)))
    assert inner_y == pytest.approx(1.103104613, abs=1e-9)
    assert outer_y == pytest.approx(2.524494115, abs=1e-9)
    assert network.in_bz(junctions[:, :2]).all()
    for y in (inner_y, outer_y, -inner_y, -outer_y):
        _assert_near(junctions, [0, y, 0], tolerance=1e-8)

    # At the same in-plane junction, the second root has cos²(kz/2)=1/72.
    # Retaining only the root attached to kz=0 would miss these branches.
    high_z = 2 * np.arccos(1 / np.sqrt(72))
    for sign in (-1, 1):
        _assert_near(points, [0, inner_y, sign * high_z], tolerance=0.04)


def test_t4_zero_retains_exact_soc_exchange_crossing():
    params = dict(network.PARAMETERS, t4=0.0)
    y = np.pi / np.sqrt(3)
    height = 2 * np.arccos(2 / 3)
    expected = np.array([[0, y, -height], [0, y, height]])
    assert np.max(_direct_middle_gap(expected, params)) < 1e-12

    curves, _ = network.trace_network(params, grid=151)
    points = _points(curves)
    assert np.max(_direct_middle_gap(points, params)) < 1e-8
    for point in expected:
        _assert_near(points, point, tolerance=0.04)


def test_nonzero_interlayer_hopping_gaps_planar_and_high_branches():
    params = dict(network.PARAMETERS, t3=0.03)
    curves, junctions = network.trace_network(params, grid=151)
    assert curves == []
    assert junctions.shape == (0, 3)

    y = 2 / np.sqrt(3) * np.arcsin(np.sqrt(2 / 3))
    high_z = 2 * np.arccos(1 / np.sqrt(72))
    former_nodes = np.array([[0, y, 0], [0, y, -high_z], [0, y, high_z]])
    gaps = _direct_middle_gap(former_nodes, params)
    np.testing.assert_allclose(gaps, [0.06, 0.007071067811865475, 0.007071067811865475],
                               atol=1e-12, rtol=1e-11)


def test_zero_out_of_plane_soc_rejects_nodal_surface_case():
    params = dict(network.PARAMETERS, lamb_z=0.0)
    with pytest.raises(ValueError, match="nodal surface"):
        network.trace_network(params, grid=151)


def test_nonzero_out_of_plane_soc_sign_preserves_nodal_points(default_network):
    curves, junctions = default_network
    points = np.concatenate((_points(curves), junctions))
    reversed_soc = dict(network.PARAMETERS, lamb_z=-network.PARAMETERS["lamb_z"])
    assert np.max(_direct_middle_gap(points, reversed_soc)) < 1e-8
