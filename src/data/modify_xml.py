"""modify_xml.py

_summary_

_extended_summary_
"""

import os
import xml.etree.ElementTree as ET

XDIR = "../../data/processed/xml_data"


def modify_xml_file(file_path, new_value):
    """
    Modify the content of specific tags in an XML file.

    Args:
    - file_path (str): Path to the XML file.
    - new_value (str): New value to set for the specified tags.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # List of tags to modify
    tags_to_modify = ['folder', 'path', 'source', 'database']

    def set_text(tag):
        for elem in root.findall(f'.//{tag}'):
            elem.text = new_value

    # Use map to apply the set_text function to each tag
    list(map(set_text, tags_to_modify))

    tree.write(file_path)  # Save the modified XML back to the file

def main(directory, new_value):
    """
    Modify XML files in a directory.

    Args:
    - directory (str): Path to the directory containing XML files.
    - new_value (str): New value to set for the specified tags in XML files.
    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        modify_xml_file(file_path, new_value)

if __name__ == "__main__":
    value_to_set = input("Enter the new value for the tags (folder, path, source, database): ")
    main(XDIR, value_to_set)
