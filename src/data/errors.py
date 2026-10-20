class DataSourceError(RuntimeError):
    """The requested source data could not be retrieved completely."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
