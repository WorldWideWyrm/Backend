from handler_interface import Handler
from chroma_storage import dataStorage

class ChromaStorageHandler(Handler):
    def __init__(self):
        super().__init__()

    def handle(self, data=None):
        print("ChromaStorageHandler is up andrunning")
        dataStorage.main()

        result = "chroma_db"

        if self.next_handler:
            return self.next_handler.handle(result)
        return result