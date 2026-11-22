"""SketchAgent tools: LLM tool classes (OpenAI function-schema style).

Each tool class follows the convention: ``definition()`` returns an OpenAI
function schema, and ``execute(args)`` takes a JSON string and returns a string.
"""
from .stroke_duplication import DetectStrokeDuplicationTool
from .stroke_ordering import OrderStrokesTool
from .compile_xml import CompileXmlTool

__all__ = ["DetectStrokeDuplicationTool", "OrderStrokesTool", "CompileXmlTool"]
