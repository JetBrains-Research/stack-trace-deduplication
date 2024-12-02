class NotFittedError(RuntimeError):
    def __init__(self, message: str = "This estimator is not fitted yet."):
        super().__init__(message)


class AlreadyFittedError(RuntimeError):
    def __init__(self, message: str = "This estimator is already fitted."):
        super().__init__(message)
