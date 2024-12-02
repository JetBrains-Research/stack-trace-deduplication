def is_logger_marker(name: str) -> bool:
    return ".openapi.diagnostic.Logger" in name


def is_slow_ops_marker(name: str) -> bool:
    return ".util.SlowOperations" in name


def is_service_marker(name: str) -> bool:
    return is_logger_marker(name) or is_slow_ops_marker(name)
