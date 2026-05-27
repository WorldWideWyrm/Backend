from abc import ABC, abstractmethod


class BaseSTTStrategy(ABC):
    @abstractmethod
    def chat(self, messeage):
        pass