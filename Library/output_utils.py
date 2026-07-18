def print_calculation_complete(label, output_path, *, artifact="Array data", copied_to=None):
    """
    Print a consistent end-of-calculation message.
    """
    print(f"\n{label} computation complete. {artifact} successfully archived at:\n > {output_path}")
    if copied_to is not None:
        print(f"Copied to temp directory:\n > {copied_to}")
