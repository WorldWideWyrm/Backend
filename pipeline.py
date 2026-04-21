from handlers.parsing_handler import RuleParsingHandler, SpellParsingHandler
from handlers.chunking_handler import RuleChunkingHandler
from handlers.storage_handler import ChromaStorageHandler

rule_parser_handler = RuleParsingHandler()
spell_parser_handler = SpellParsingHandler()
rule_chunking_handler = RuleChunkingHandler()
storage_handler = ChromaStorageHandler()

rule_parser_handler.set_next(spell_parser_handler).set_next(rule_chunking_handler).set_next(storage_handler)

rule_parser_handler.handle()