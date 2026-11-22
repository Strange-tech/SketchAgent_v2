"""Compile / validate the LLM's <strokes> XML before use.

The trunk agent's output is frequently malformed (truncated, missing markers,
wrong nesting, bad coordinates). This tool checks the XML and returns a specific
error message when it is invalid, so the error can be fed back to the LLM for a
repair pass instead of silently dropping the strokes.
"""
import json
import re
import xml.etree.ElementTree as ET


class CompileXmlTool:
    def definition(self):
        return {
            "name": "compile",
            "description": (
                "Validate a <strokes> XML string. Returns valid=true, or valid=false "
                "with a specific error describing what is wrong so the LLM can fix it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "xml": {"type": "string", "description": "The <strokes> XML text to validate."},
                },
                "required": ["xml"],
            },
        }

    def execute(self, args):
        try:
            params = json.loads(args)
        except json.JSONDecodeError as e:
            return json.dumps({"valid": False, "error": "arguments are not valid JSON: " + str(e)})
        return json.dumps(self.validate(params.get("xml", "") or ""))

    def validate(self, xml_text):
        """Return {"valid": True} or {"valid": False, "error": "..."}."""
        if not (xml_text or "").strip():
            return {"valid": False, "error": "empty output"}

        start = xml_text.find("<strokes>")
        end = xml_text.find("</strokes>")
        if start == -1 or end == -1:
            return {"valid": False, "error": "missing <strokes>...</strokes> block"}
        block = xml_text[start:end + len("</strokes>")].strip()

        try:
            root = ET.fromstring(f"<wrap>{block}</wrap>")
        except ET.ParseError as e:
            return {"valid": False, "error": "XML parse error: " + str(e)}

        strokes = root.find("strokes")
        if strokes is None:
            return {"valid": False, "error": "no <strokes> element found"}

        children = list(strokes)
        if not children:
            return {"valid": False, "error": "<strokes> has no <sN> stroke children"}

        for s in children:
            if not re.match(r"^s\d+$", s.tag):
                return {"valid": False, "error": f"unexpected element <{s.tag}>; expected <sN>"}

            points_el = s.find("points")
            if points_el is None or not (points_el.text or "").strip():
                return {"valid": False, "error": f"<{s.tag}> is missing a non-empty <points> child"}

            t_values_el = s.find("t_values")
            if t_values_el is None or not (t_values_el.text or "").strip():
                return {"valid": False, "error": f"<{s.tag}> is missing a non-empty <t_values> child"}

            for raw in (points_el.text or "").split(","):
                p = raw.strip().strip("'\"")
                if not p:
                    continue
                if not re.match(r"^x\d+y\d+$", p):
                    return {"valid": False, "error": f"<{s.tag}> has invalid point '{raw.strip()}'; expected 'xNyM'"}

        return {"valid": True}
