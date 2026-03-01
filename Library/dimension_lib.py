

def map_k_by_order(ki_ij, kj_ij, kk, order: str):
    """
    order is a permutation of 'xyz' specifying which variable gets:
      - ki_ij (grid i,j)
      - kj_ij (grid i,j)
      - kk    (fixed third axis)

    Convention:
      - first letter of order gets ki_ij
      - second letter gets kj_ij
      - third letter gets kk

    Examples:
      order='xyz' -> kx=ki_ij, ky=kj_ij, kz=kk
      order='yzx' -> ky=ki_ij, kz=kj_ij, kx=kk
      order='xzy' -> kx=ki_ij, kz=kj_ij, ky=kk
    """
    if not isinstance(order, str):
        raise TypeError("order must be a string like 'xyz'")
    order = order.lower()
    if sorted(order) != ['x', 'y', 'z']:
        raise ValueError("order must be a permutation of 'xyz' (e.g. 'xyz', 'yzx', 'xzy', etc.)")

    # assign slots (first->ki, second->kj, third->kk)
    slot = {order[0]: ki_ij, order[1]: kj_ij, order[2]: kk}

    return slot['x'], slot['y'], slot['z']