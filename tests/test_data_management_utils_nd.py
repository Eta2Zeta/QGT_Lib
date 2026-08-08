import numpy as np
import pytest

from Library.data_management_utils_nd import build_parameter_points


def _parameter_axis(parameter_range, count, scale="log"):
    _, _, axes, shape = build_parameter_points(
        {"parameter": parameter_range},
        {"parameter": {"n": count, "scale": scale}},
    )
    assert shape == (count,)
    return axes[0]


def test_symmetric_log_spacing_is_mirrored_and_concentrated_near_zero():
    axis = _parameter_axis((-0.5, 0.5), 11)
    expected_positive = np.geomspace(0.0005, 0.5, 5)

    np.testing.assert_allclose(axis, -axis[::-1])
    np.testing.assert_allclose(axis[6:], expected_positive)
    assert axis[5] == 0.0
    assert axis[0] == -0.5
    assert axis[-1] == 0.5


def test_even_symmetric_log_spacing_remains_mirrored_without_zero():
    axis = _parameter_axis((-0.5, 0.5), 10)

    np.testing.assert_allclose(axis, -axis[::-1])
    assert not np.any(axis == 0.0)


def test_positive_log_spacing_is_unchanged():
    axis = _parameter_axis((0.001, 1.0), 4)

    np.testing.assert_allclose(axis, np.logspace(-3.0, 0.0, 4))


def test_asymmetric_zero_crossing_log_range_is_rejected():
    with pytest.raises(ValueError, match="symmetric zero-centered range"):
        _parameter_axis((-0.25, 0.5), 11)
