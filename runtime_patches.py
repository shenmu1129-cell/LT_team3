"""Project-local runtime compatibility patches."""


def apply():
    """Apply local patches without changing the shared conda environment."""
    try:
        import importlib.metadata as metadata
    except Exception:
        return

    original_version = metadata.version

    def version(package_name):
        value = original_version(package_name)
        if package_name.lower() == "tqdm" and value is None:
            return "4.65.2"
        return value

    metadata.version = version
