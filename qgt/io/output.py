def print_calculation_complete(label, output_path, *, artifact="Array data"):
    """
    Print a consistent end-of-calculation message.
    """
    print(f"\n{label} computation complete. {artifact} successfully archived at:\n > {output_path}")
