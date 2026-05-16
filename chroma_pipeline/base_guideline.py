from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunking(self, pages: list[dict])-> list[dict]:
        pass

    @abstractmethod
    def normalize(self, chunks: list[dict])-> list[dict]:
        pass

    def clean_metadata(self, metadata: dict) -> dict:
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
