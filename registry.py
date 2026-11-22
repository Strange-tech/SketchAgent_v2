"""Tool registry (referencing SEAL/BaseAgent's registry.py)."""


class Registry:
    def __init__(self, *tools):
        self.tools = {}
        for t in tools:
            self.tools[t.definition()["name"]] = t

    def definitions(self):
        """OpenAI function-schema list for the registered tools."""
        return [{"type": "function", "function": t.definition()} for t in self.tools.values()]

    def execute(self, name, args):
        t = self.tools.get(name)
        if t is None:
            return "Error: Unknown tool " + name
        return t.execute(args)
