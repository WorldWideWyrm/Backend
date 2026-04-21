from base_handler import Handler
from Parsers import rule_parser, spell_parser

class RuleParsingHandler(Handler):
    def __init__(self):
        super().__init__()

    def handle(self, data=None):
        print("RuleParsingHandler is running")
        rule_parser.main()

        result = "pdfs/rule_glossary.txt"

        if self.next_handler:
            return self.next_handler.handle(result)
        return result


class SpellParsingHandler(Handler):
    def __init__(self):
        super().__init__()

    def handle(self, data=None):
        print("SpellParsingHandler is running")
        spell_parser.main()

        result = "pdfs/spell_chunks.json"

        if self.next_handler:
            return self.next_handler.handle(result)
        return result