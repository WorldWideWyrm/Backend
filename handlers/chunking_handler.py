from base_handler import Handler
from Chunking import rule_chunking

class RuleChunkingHandler(Handler):
    def __init__(self):
        super().__init__()

    def handle(self, data=None):
        print("RuleChunkingHandler is running")

        rule_chunking.main()

        result = "pdfs/rules_chunks.json"

        if self.next_handler:
            return self.next_handler.handle(result)
        return result