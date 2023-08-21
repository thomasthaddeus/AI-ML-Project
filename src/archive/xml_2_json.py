"""
_summary_

_extended_summary_

Returns:
    _type_: _description_
"""

import json
import xml.etree.ElementTree as ET

def xml_to_dict(element):
    """Convert an XML element to a dictionary."""
    data = {}
    for child in element:
        data[child.tag] = xml_to_dict(child) if len(child) else child.text
    return data

def xml_file_to_json(xml_file, json_file):
    """Convert an XML file to a JSON file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = xml_to_dict(root)

    with open(file=json_file, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Usage
xml_file_to_json('input.xml', 'output.json')
