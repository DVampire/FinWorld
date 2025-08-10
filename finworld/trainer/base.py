class Trainer():
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}()"

    def __repr__(self):
        return self.__str__()

    def train(self, **kwargs):
        raise NotImplementedError("The train method should be implemented in the subclass.")

    def valid(self, **kwargs):
        raise NotImplementedError("The valid method should be implemented in the subclass.")

    def test(self, **kwargs):
        raise NotImplementedError("The test method should be implemented in the subclass.")
