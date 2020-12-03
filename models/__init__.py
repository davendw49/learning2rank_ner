from abc import ABCMeta, abstractmethod
import pkgutil as _pkgutil
import os


class BaseModel(object, metaclass=ABCMeta):
    def __init__(self, training_data, n_feature, h1_units, h2_units, epoch, lr, number_of_trees=10, plot=True):
        self.logger = None

    def fit(self, k):
        """
        fit the data via training
        """

    def predict(self, data):
        """
        predict data
        """

    def validate(self, data, k):
        """
        validate topk
        """


for _, _modname, _ in _pkgutil.walk_packages(path=__path__, prefix=__name__ + "."):
    __import__(_modname)