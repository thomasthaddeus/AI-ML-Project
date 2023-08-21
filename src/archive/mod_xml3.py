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
