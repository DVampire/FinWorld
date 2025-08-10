class Metric():
    """
    Base class for all metrics.
    """
    def __init__(self, **kwargs):
        super(Metric, self).__init__()

    def __call__(self, **kwargs):
        """
        Call the metric with the given arguments.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        class_name = self.__class__.__name__
        params_str = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_'))
        str = f"{class_name}({params_str})"
        return str

    def __repr__(self):
        return self.__str__()
