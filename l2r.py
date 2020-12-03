import logging

from models import BaseModel


class Rank:
    def __init__(self, rank_model='lambda_rank', **kwargs):
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

