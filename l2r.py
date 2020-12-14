import logging

from models import BaseModel


class Rank:
    def __init__(self, rank_model='lambdarank', **kwargs):
        self.logger = logging.getLogger('Learning to Rank')

        self.rank_model = rank_model
        self.handler = self._load_handler(self.rank_model)
        self.handler = self.handler(**kwargs) if self.handler else None

    def _load_handler(self, rank_model):
        for cls in BaseModel.__subclasses__():
            if cls.__name__.lower() == rank_model.lower():
                return cls
        self.logger.error(f'rank model {rank_model} not found')
        return None

    def fit(self, k):
        if self.handler is None:
            self.logger.error(f'model {self.rank_model} load failed, please try to re-construct.')

        self.logger.info(f'start to use {self.rank_model} to fit the training data.')
        self.handler.fit(k)
        self.logger.info(f'finish fitting')

    def predict(self, data, k):
        if self.handler is None:
            self.logger.error(f'model {self.rank_model} load failed, please try to re-construct.')

        self.logger.info(f'start to use {self.rank_model} to predict the test data.')
        predicted_scores, predicted_scores_aeid = self.handler.predict(data, k)
        self.logger.info(f'finish predict')
        return predicted_scores, predicted_scores_aeid

    def validate(self, data, k):
        if self.handler is None:
            self.logger.error(f'model {self.rank_model} load failed, please try to re-construct.')

        self.logger.info(f'start to use {self.rank_model} to validate the test data.')
        ndcg_k_list = self.handler.validate(data, k)
        self.logger.info(f'finish validate')
        return ndcg_k_list