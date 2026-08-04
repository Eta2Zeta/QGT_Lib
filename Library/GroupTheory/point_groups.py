"""Point-group elements, operations, and character tables.

The matrices stored here act on Cartesian polar vectors, including momentum,
according to ``v_transformed = R @ v``.  Transformations of axial-vector
components, such as Berry curvature, are deliberately not folded into these
coordinate-space operations.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class PointGroupElement:
    """One named point-group operation in a Cartesian coordinate convention."""

    name: str
    conjugacy_class: str
    _matrix_entries: tuple[tuple[float, float, float], ...] = field(repr=False)

    def __init__(self, name, conjugacy_class, matrix):
        matrix_array = np.asarray(matrix, dtype=float)
        if matrix_array.shape != (3, 3):
            raise ValueError("A point-group operation must be a 3x3 matrix.")
        if not np.allclose(matrix_array.T @ matrix_array, np.eye(3), atol=1e-12):
            raise ValueError("A point-group operation must be orthogonal.")

        determinant = float(np.linalg.det(matrix_array))
        if not np.isclose(abs(determinant), 1.0, atol=1e-12):
            raise ValueError("A point-group operation must have determinant +1 or -1.")

        entries = tuple(tuple(float(value) for value in row) for row in matrix_array)
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "conjugacy_class", str(conjugacy_class))
        object.__setattr__(self, "_matrix_entries", entries)

    @property
    def matrix(self):
        """Return the operation's 3x3 Cartesian matrix."""
        return np.asarray(self._matrix_entries, dtype=float)

    @property
    def inverse_matrix(self):
        """Return the inverse operation matrix."""
        return self.matrix.T

    @property
    def determinant(self):
        """Return +1 for proper and -1 for improper operations."""
        return int(round(np.linalg.det(self.matrix)))

    @property
    def is_proper(self):
        """Return whether this is a proper rotation."""
        return self.determinant == 1


class PointGroup:
    """Finite three-dimensional point group with a character table."""

    def __init__(self, name, elements, character_table):
        self.name = str(name)
        self._elements = tuple(elements)

        if not self._elements:
            raise ValueError("A point group must contain at least one element.")

        element_names = [element.name for element in self._elements]
        if len(element_names) != len(set(element_names)):
            raise ValueError("Point-group element names must be unique.")
        self._elements_by_name = MappingProxyType(
            {element.name: element for element in self._elements}
        )

        classes = {}
        for element in self._elements:
            classes.setdefault(element.conjugacy_class, []).append(element)
        self._conjugacy_classes = MappingProxyType(
            {label: tuple(members) for label, members in classes.items()}
        )

        class_labels = set(self._conjugacy_classes)
        table = {}
        for irrep, row in character_table.items():
            row = dict(row)
            if set(row) != class_labels:
                missing = sorted(class_labels - set(row))
                extra = sorted(set(row) - class_labels)
                raise ValueError(
                    f"Character row {irrep!r} has incorrect classes; "
                    f"missing={missing}, extra={extra}."
                )
            table[str(irrep)] = MappingProxyType(row)

        if len(table) != len(class_labels):
            raise ValueError(
                "A complete character table must have one irrep per conjugacy class."
            )

        self._character_table = MappingProxyType(table)
        self._identity = self._find_identity()
        self._validate_character_table()

    @property
    def order(self):
        """Number of elements in the group."""
        return len(self._elements)

    @property
    def elements(self):
        """All operations, in the order used for character projection sums."""
        return self._elements

    @property
    def element_names(self):
        return tuple(element.name for element in self._elements)

    @property
    def identity(self):
        return self._identity

    @property
    def conjugacy_classes(self):
        """Read-only mapping from class labels to their member operations."""
        return self._conjugacy_classes

    @property
    def class_sizes(self):
        return MappingProxyType(
            {
                label: len(members)
                for label, members in self._conjugacy_classes.items()
            }
        )

    @property
    def irreps(self):
        return tuple(self._character_table)

    @property
    def character_table(self):
        """Read-only mapping ``irrep -> conjugacy class -> character``."""
        return self._character_table

    def element(self, name):
        """Return a group element by its exact canonical name."""
        try:
            return self._elements_by_name[name]
        except KeyError as error:
            valid = ", ".join(self.element_names)
            raise KeyError(f"Unknown {self.name} element {name!r}. Valid: {valid}.") from error

    def conjugacy_class(self, label):
        """Return all operations belonging to one conjugacy class."""
        try:
            return self._conjugacy_classes[label]
        except KeyError as error:
            valid = ", ".join(self._conjugacy_classes)
            raise KeyError(
                f"Unknown {self.name} conjugacy class {label!r}. Valid: {valid}."
            ) from error

    def character(self, irrep, element):
        """Return ``chi_irrep(element)`` for an element name or object."""
        row = self._character_row(irrep)
        if isinstance(element, str):
            element = self.element(element)
        elif element not in self._elements:
            raise ValueError(f"The supplied element does not belong to {self.name}.")
        return row[element.conjugacy_class]

    def class_character(self, irrep, conjugacy_class):
        """Return an irrep character using a conjugacy-class label."""
        row = self._character_row(irrep)
        if conjugacy_class not in self._conjugacy_classes:
            valid = ", ".join(self._conjugacy_classes)
            raise KeyError(
                f"Unknown {self.name} conjugacy class {conjugacy_class!r}. "
                f"Valid: {valid}."
            )
        return row[conjugacy_class]

    def characters(self, irrep):
        """Return characters for all elements in ``self.elements`` order."""
        return tuple(self.character(irrep, element) for element in self._elements)

    def characters_by_element(self, irrep):
        """Return a read-only ``element name -> character`` mapping."""
        return MappingProxyType(
            {
                element.name: self.character(irrep, element)
                for element in self._elements
            }
        )

    def irrep_dimension(self, irrep):
        """Return the dimension of an irreducible representation."""
        return int(round(self.character(irrep, self.identity).real))

    def compose(self, left, right):
        """Return ``left * right``, with ``right`` acting first."""
        left = self._coerce_element(left)
        right = self._coerce_element(right)
        return self._match_matrix(left.matrix @ right.matrix)

    def inverse(self, element):
        """Return the inverse of a group element."""
        element = self._coerce_element(element)
        return self._match_matrix(element.inverse_matrix)

    def transform_k(self, element, k_points, *, inverse=False):
        """Apply a group operation to one or more Cartesian momentum points.

        Parameters
        ----------
        element : str or PointGroupElement
            Operation to apply.
        k_points : array_like
            A single ``(kx, ky, kz)`` point with shape ``(3,)``, or any array
            whose final axis contains Cartesian momentum components.
        inverse : bool, default=False
            Apply ``R_g^{-1}`` instead of ``R_g``. Character projection uses
            ``inverse=True`` to construct ``f(R_g^{-1} k)``.
        """
        element = self._coerce_element(element)
        k_points = np.asarray(k_points, dtype=float)
        if k_points.ndim == 0 or k_points.shape[-1] != 3:
            raise ValueError(
                "k_points must have shape (3,) or (..., 3), with Cartesian "
                "components on the final axis."
            )

        operation = element.inverse_matrix if inverse else element.matrix
        return np.einsum("ij,...j->...i", operation, k_points)

    def project_onto_irrep(self, transformed_data, irrep, *, element_axis=0):
        r"""Project a complete group orbit of data onto one irrep.

        ``transformed_data`` must contain the already-evaluated copies

        .. math::

            F_g(\mathbf{k}) = f(R_g^{-1}\mathbf{k})

        along ``element_axis``, in exactly the order given by
        ``self.elements``. The returned array is

        .. math::

            (P^\Gamma f)(\mathbf{k}) =
            \frac{d_\Gamma}{|G|}
            \sum_{g\in G}\chi_\Gamma(g)^* F_g(\mathbf{k}).

        The remaining axes may describe a scalar grid, vector components,
        bands, or any other data on which the character sum acts
        componentwise.
        """
        transformed_data = np.asarray(transformed_data)
        if transformed_data.ndim == 0:
            raise ValueError("transformed_data must have at least one axis.")

        element_axis = int(element_axis)
        if element_axis < 0:
            element_axis += transformed_data.ndim
        if not 0 <= element_axis < transformed_data.ndim:
            raise ValueError(
                f"element_axis={element_axis} is invalid for data with "
                f"{transformed_data.ndim} dimensions."
            )
        if transformed_data.shape[element_axis] != self.order:
            raise ValueError(
                "The group-element axis of transformed_data must have length "
                f"{self.order}; received {transformed_data.shape[element_axis]}."
            )

        data_by_element = np.moveaxis(transformed_data, element_axis, 0)
        characters = np.asarray(self.characters(irrep))
        irrep_dimension = self.irrep_dimension(irrep)
        character_sum = np.tensordot(
            np.conjugate(characters),
            data_by_element,
            axes=(0, 0),
        )
        return (irrep_dimension / self.order) * character_sum

    def _coerce_element(self, element):
        if isinstance(element, str):
            return self.element(element)
        if element not in self._elements:
            raise ValueError(f"The supplied element does not belong to {self.name}.")
        return element

    def _match_matrix(self, matrix):
        matches = [
            element
            for element in self._elements
            if np.allclose(element.matrix, matrix, atol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"The stored operations for {self.name} are not closed and unique."
            )
        return matches[0]

    def _find_identity(self):
        matches = [
            element
            for element in self._elements
            if np.allclose(element.matrix, np.eye(3), atol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError("A point group must contain exactly one identity matrix.")
        return matches[0]

    def _character_row(self, irrep):
        try:
            return self._character_table[irrep]
        except KeyError as error:
            valid = ", ".join(self.irreps)
            raise KeyError(f"Unknown {self.name} irrep {irrep!r}. Valid: {valid}.") from error

    def _validate_character_table(self):
        dimensions = [self.irrep_dimension(irrep) for irrep in self.irreps]
        if sum(dimension**2 for dimension in dimensions) != self.order:
            raise ValueError("Irrep dimensions do not satisfy the group-order sum rule.")

        for irrep_a in self.irreps:
            for irrep_b in self.irreps:
                inner_product = sum(
                    len(elements)
                    * np.conjugate(
                        self.class_character(irrep_a, conjugacy_class)
                    )
                    * self.class_character(irrep_b, conjugacy_class)
                    for conjugacy_class, elements in self.conjugacy_classes.items()
                )
                expected = self.order if irrep_a == irrep_b else 0
                if not np.isclose(inner_product, expected, atol=1e-12):
                    raise ValueError(
                        f"Character rows {irrep_a!r} and {irrep_b!r} "
                        "fail orthogonality."
                    )


def _rotation_about_z(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _twofold_rotation_in_xy_plane(axis_angle):
    axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
    return 2.0 * np.outer(axis, axis) - np.eye(3)


class D3dPointGroup(PointGroup):
    """The order-12 ``D3d`` point group in a fixed Cartesian convention.

    ``C3+`` rotates by ``+2*pi/3`` about ``+z``. ``C2p_0`` is a twofold
    rotation about ``+x``; the other two twofold axes are obtained by rotating
    that axis by ``2*pi/3`` and ``4*pi/3``.  Each ``sigma_d_n`` is inversion
    times ``C2p_n`` and therefore has its plane normal to the corresponding
    twofold axis.
    """

    def __init__(self):
        identity = np.eye(3)
        inversion = -identity
        c3_plus = _rotation_about_z(2.0 * np.pi / 3.0)
        c3_minus = _rotation_about_z(-2.0 * np.pi / 3.0)
        c2_operations = tuple(
            _twofold_rotation_in_xy_plane(2.0 * np.pi * index / 3.0)
            for index in range(3)
        )

        elements = (
            PointGroupElement("E", "E", identity),
            PointGroupElement("C3+", "2C3", c3_plus),
            PointGroupElement("C3-", "2C3", c3_minus),
            *(
                PointGroupElement(f"C2p_{index}", "3C2p", operation)
                for index, operation in enumerate(c2_operations)
            ),
            PointGroupElement("i", "i", inversion),
            PointGroupElement("S6+", "2S6", inversion @ c3_minus),
            PointGroupElement("S6-", "2S6", inversion @ c3_plus),
            *(
                PointGroupElement(
                    f"sigma_d_{index}",
                    "3sigma_d",
                    inversion @ operation,
                )
                for index, operation in enumerate(c2_operations)
            ),
        )

        character_table = {
            "A1g": {
                "E": 1,
                "2C3": 1,
                "3C2p": 1,
                "i": 1,
                "2S6": 1,
                "3sigma_d": 1,
            },
            "A2g": {
                "E": 1,
                "2C3": 1,
                "3C2p": -1,
                "i": 1,
                "2S6": 1,
                "3sigma_d": -1,
            },
            "Eg": {
                "E": 2,
                "2C3": -1,
                "3C2p": 0,
                "i": 2,
                "2S6": -1,
                "3sigma_d": 0,
            },
            "A1u": {
                "E": 1,
                "2C3": 1,
                "3C2p": 1,
                "i": -1,
                "2S6": -1,
                "3sigma_d": -1,
            },
            "A2u": {
                "E": 1,
                "2C3": 1,
                "3C2p": -1,
                "i": -1,
                "2S6": -1,
                "3sigma_d": 1,
            },
            "Eu": {
                "E": 2,
                "2C3": -1,
                "3C2p": 0,
                "i": -2,
                "2S6": 1,
                "3sigma_d": 0,
            },
        }

        super().__init__("D3d", elements, character_table)
