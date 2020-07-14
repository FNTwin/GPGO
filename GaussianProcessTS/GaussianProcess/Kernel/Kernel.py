
class Kernel:
    """
    Parent class for kernels. It's used to check if the type of every custom kernel is right.
    """

    def __init__(self, *args, **kwargs):
        self.__type = "Kernel"

    def gettype(self):
        return self.__type



