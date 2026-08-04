import os

import numpy as np
import pytest

from Calc_QGT_2D_D3d_Projection import (
    _project_all_irreps_parallel,
    _project_irrep_worker,
)
from Library.GroupTheory import D3dPointGroup


def test_irrep_worker_matches_serial_projection(tmp_path):
    group = D3dPointGroup()
    berry_xy_by_group = np.arange(
        group.order * 2 * 3 * 4 * 2,
        dtype=float,
    ).reshape(group.order, 2, 3, 4, 2)
    berry_path = tmp_path / "berry.npy"
    output_path = tmp_path / "projected.npy"
    np.save(berry_path, berry_xy_by_group)

    irrep, returned_path, worker_pid = _project_irrep_worker(
        "Eg",
        group.characters("Eg"),
        group.irrep_dimension("Eg"),
        group.order,
        str(berry_path),
        str(output_path),
    )

    assert irrep == "Eg"
    assert returned_path == str(output_path)
    assert worker_pid == os.getpid()
    assert np.allclose(
        np.load(output_path),
        group.project_onto_irrep(berry_xy_by_group, "Eg"),
    )


def test_parallel_projection_uses_one_process_per_irrep(tmp_path):
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except PermissionError:
        pytest.skip("The execution sandbox blocks multiprocessing semaphores.")

    group = D3dPointGroup()
    berry_xy_by_group = np.arange(
        group.order * 2 * 3 * 4 * 2,
        dtype=float,
    ).reshape(group.order, 2, 3, 4, 2)

    projected, worker_pids = _project_all_irreps_parallel(
        berry_xy_by_group,
        group,
        str(tmp_path),
    )

    expected = np.stack(
        [
            group.project_onto_irrep(berry_xy_by_group, irrep)
            for irrep in group.irreps
        ],
        axis=0,
    )
    assert np.allclose(projected, expected)
    assert tuple(worker_pids) == group.irreps
    assert len(set(worker_pids.values())) == len(group.irreps)
