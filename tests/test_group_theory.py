import numpy as np
import pytest

from Library.GroupTheory import D3dPointGroup, PointGroupElement


@pytest.fixture
def d3d():
    return D3dPointGroup()


def test_d3d_has_expected_order_classes_and_irreps(d3d):
    assert d3d.name == "D3d"
    assert d3d.order == 12
    assert d3d.class_sizes == {
        "E": 1,
        "2C3": 2,
        "3C2p": 3,
        "i": 1,
        "2S6": 2,
        "3sigma_d": 3,
    }
    assert d3d.irreps == ("A1g", "A2g", "Eg", "A1u", "A2u", "Eu")
    assert [d3d.irrep_dimension(irrep) for irrep in d3d.irreps] == [
        1,
        1,
        2,
        1,
        1,
        2,
    ]


def test_d3d_character_lookup_accepts_elements_and_names(d3d):
    assert d3d.character("Eg", "E") == 2
    assert d3d.character("Eg", "C3+") == -1
    assert d3d.character("Eg", d3d.element("C2p_2")) == 0
    assert d3d.class_character("A2u", "3sigma_d") == 1

    eg_by_element = d3d.characters_by_element("Eg")
    assert eg_by_element["C3-"] == -1
    assert eg_by_element["i"] == 2
    assert d3d.characters("Eg") == tuple(
        eg_by_element[name] for name in d3d.element_names
    )


def test_d3d_character_table_matches_the_standard_table(d3d):
    expected_rows = {
        "A1g": (1, 1, 1, 1, 1, 1),
        "A2g": (1, 1, -1, 1, 1, -1),
        "Eg": (2, -1, 0, 2, -1, 0),
        "A1u": (1, 1, 1, -1, -1, -1),
        "A2u": (1, 1, -1, -1, -1, 1),
        "Eu": (2, -1, 0, -2, 1, 0),
    }
    class_order = ("E", "2C3", "3C2p", "i", "2S6", "3sigma_d")

    for irrep, expected in expected_rows.items():
        actual = tuple(
            d3d.class_character(irrep, conjugacy_class)
            for conjugacy_class in class_order
        )
        assert actual == expected


def test_d3d_coordinate_matrices_use_documented_convention(d3d):
    x_hat = np.array([1.0, 0.0, 0.0])
    vector = np.array([1.0, 2.0, 3.0])

    assert np.allclose(
        d3d.element("C3+").matrix @ x_hat,
        [-0.5, np.sqrt(3.0) / 2.0, 0.0],
    )
    assert np.allclose(d3d.element("C2p_0").matrix @ vector, [1.0, -2.0, -3.0])
    assert np.allclose(d3d.element("sigma_d_0").matrix @ vector, [-1.0, 2.0, 3.0])
    assert np.allclose(d3d.element("i").matrix @ vector, -vector)


def test_d3d_operations_are_orthogonal_and_have_expected_parity(d3d):
    for element in d3d.elements:
        assert np.allclose(element.matrix.T @ element.matrix, np.eye(3))

    assert all(element.is_proper for element in d3d.elements[:6])
    assert all(not element.is_proper for element in d3d.elements[6:])


def test_d3d_is_closed_and_each_element_has_an_inverse(d3d):
    for left in d3d.elements:
        for right in d3d.elements:
            assert d3d.compose(left, right) in d3d.elements

    for element in d3d.elements:
        inverse = d3d.inverse(element)
        assert d3d.compose(element, inverse) == d3d.identity
        assert d3d.compose(inverse, element) == d3d.identity

    assert d3d.compose("C3+", "C3+") == d3d.element("C3-")
    assert d3d.inverse("C3+") == d3d.element("C3-")


def test_transform_k_applies_forward_and_inverse_operations(d3d):
    k_point = np.array([1.2, -0.7, 0.4])

    transformed = d3d.transform_k("C3+", k_point)
    inverse_transformed = d3d.transform_k("C3+", k_point, inverse=True)

    assert np.allclose(transformed, d3d.element("C3+").matrix @ k_point)
    assert np.allclose(
        inverse_transformed,
        d3d.element("C3+").inverse_matrix @ k_point,
    )
    assert np.allclose(
        d3d.transform_k("C3+", transformed, inverse=True),
        k_point,
    )


def test_transform_k_supports_arrays_with_components_on_last_axis(d3d):
    kx, ky = np.meshgrid(
        np.array([-1.0, 1.0]),
        np.array([-2.0, 0.0, 2.0]),
    )
    k_points = np.stack((kx, ky, np.full_like(kx, 0.5)), axis=-1)

    transformed = d3d.transform_k("i", k_points, inverse=True)

    assert transformed.shape == k_points.shape
    assert np.allclose(transformed, -k_points)


@pytest.mark.parametrize(
    ("irrep", "function"),
    [
        ("A1g", lambda k: np.ones(k.shape[:-1])),
        ("A2u", lambda k: k[..., 2]),
        ("Eg", lambda k: k[..., 0] ** 2 - k[..., 1] ** 2),
    ],
)
def test_character_projector_reproduces_known_d3d_basis_functions(
    d3d,
    irrep,
    function,
):
    k_point = np.array([0.37, -0.22, 0.51])
    expected = function(k_point)
    transformed_data = np.stack(
        [
            function(d3d.transform_k(element, k_point, inverse=True))
            for element in d3d.elements
        ]
    )

    assert np.allclose(
        d3d.project_onto_irrep(transformed_data, irrep),
        expected,
        atol=1e-12,
    )

    for other_irrep in set(d3d.irreps) - {irrep}:
        assert np.allclose(
            d3d.project_onto_irrep(transformed_data, other_irrep),
            0.0,
            atol=1e-12,
        )


def test_character_projector_extracts_each_part_of_a_mixed_function(d3d):
    k_point = np.array([0.37, -0.22, 0.51])

    def mixed_function(k):
        return 2.0 + k[..., 2] + k[..., 0] ** 2 - k[..., 1] ** 2

    transformed_data = np.stack(
        [
            mixed_function(d3d.transform_k(element, k_point, inverse=True))
            for element in d3d.elements
        ]
    )

    assert np.allclose(
        d3d.project_onto_irrep(transformed_data, "A1g"),
        2.0,
    )
    assert np.allclose(
        d3d.project_onto_irrep(transformed_data, "A2u"),
        k_point[2],
    )
    assert np.allclose(
        d3d.project_onto_irrep(transformed_data, "Eg"),
        k_point[0] ** 2 - k_point[1] ** 2,
    )


def test_character_projector_supports_batched_k_points(d3d):
    k_points = np.array(
        [
            [0.1, 0.2, 0.3],
            [-0.4, 0.5, -0.6],
        ]
    )
    z_function = lambda k: k[..., 2]
    transformed_data = np.stack(
        [
            z_function(d3d.transform_k(element, k_points, inverse=True))
            for element in d3d.elements
        ],
        axis=-1,
    )

    projected = d3d.project_onto_irrep(
        transformed_data,
        "A2u",
        element_axis=-1,
    )

    assert projected.shape == (2,)
    assert np.allclose(projected, k_points[:, 2])


def test_invalid_group_queries_raise_clear_errors(d3d):
    with pytest.raises(KeyError, match="Unknown D3d element"):
        d3d.element("C4")
    with pytest.raises(KeyError, match="Unknown D3d irrep"):
        d3d.character("T2g", "E")
    with pytest.raises(KeyError, match="Unknown D3d conjugacy class"):
        d3d.class_character("Eg", "3C4")
    with pytest.raises(ValueError, match="k_points must have shape"):
        d3d.transform_k("E", [1.0, 2.0])
    with pytest.raises(ValueError, match="must have length 12"):
        d3d.project_onto_irrep(np.zeros((11, 4, 4)), "A1g")


def test_point_group_element_rejects_nonorthogonal_matrix():
    with pytest.raises(ValueError, match="orthogonal"):
        PointGroupElement("bad", "bad", np.ones((3, 3)))
