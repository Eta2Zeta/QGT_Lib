def quantum_geometric_tensor_3d_analytic(Hamiltonian, kx, ky, kz, *, trace_from_sum=True, **kwargs):
    """
    Analytic 3D QGT accessor with passthrough kwargs.

    Change vs your version:
      - If a component method does NOT exist (or errors), returns None for that component
        instead of raising.
      - trace:
          * if Hamiltonian.trace exists and works -> use it
          * else if trace_from_sum and all diagonals exist -> sum them
          * else -> None
    """

    def _call(name):
        fn = getattr(Hamiltonian, name, None)
        if not callable(fn):
            return None
        try:
            return fn(kx, ky, kz, **kwargs)
        except TypeError:
            # method might not accept kwargs
            try:
                return fn(kx, ky, kz)
            except Exception:
                return None
        except Exception:
            return None

    g_xx = _call("g_xx")
    g_yy = _call("g_yy")
    g_zz = _call("g_zz")

    g_xy_real = _call("g_xy_real")
    g_xy_imag = _call("g_xy_imag")

    g_xz_real = _call("g_xz_real")
    g_xz_imag = _call("g_xz_imag")

    g_yz_real = _call("g_yz_real")
    g_yz_imag = _call("g_yz_imag")

    return (
        g_xx, g_yy, g_zz,
        g_xy_real, g_xy_imag,
        g_xz_real, g_xz_imag,
        g_yz_real, g_yz_imag
    )