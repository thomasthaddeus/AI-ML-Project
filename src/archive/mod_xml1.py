import os
import xml.etree.ElementTree as ET

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

    # Search for the tag in the entire XML tree
    for tag in tags_to_modify:
        for elem in root.findall(f'.//{tag}'):
            elem.text = new_value

    tree.write(file_path)  # Save the modified XML back to the file

def main(directory, new_value):
    """
    Recursively search for XML files in a directory and modify them.

    Args:
    - directory (str): Path to the directory to start the search.
    - new_value (str): New value to set for the specified tags in XML files.
    """
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(subdir, file)
                modify_xml_file(file_path, new_value)

if __name__ == "__main__":
    directory_to_search = input("Enter the directory path to search for XML files: ")
    value_to_set = input("Enter the new value for the tags (folder, path, source, database): ")
    main(directory_to_search, value_to_set)
