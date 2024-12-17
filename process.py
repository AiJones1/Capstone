# html_processor.py
import re
import webbrowser
from tkinter import Tk, filedialog

def convert_htm_to_html(file_path):
    """
    Convert .htm file to .html by renaming the file.
    """
    if file_path.endswith('.htm'):
        new_path = file_path + 'l'  
        try:
            with open(file_path, 'r', encoding='utf-8') as infile, open(new_path, 'w', encoding='utf-8') as outfile:
                outfile.write(infile.read())
            print(f"Converted .htm to .html: {new_path}")
            return new_path
        except Exception as e:
            print(f"Error converting .htm to .html: {e}")
            return None
    else:
        print("File is already .html or invalid.")
        return file_path

def replace_line_directly(html_content, old_line, new_line):
    print(f"Replacing:\nOLD: {old_line}\nNEW: {new_line}")
    old_line_escaped = re.escape(old_line)
    adjusted_content = re.sub(old_line_escaped, new_line, html_content)
    return adjusted_content

def handle_hierarchical_conflict(html_content):
    """
    Adjust lines with double hierarchies, splitting them and adding padding-left dynamically
    to subsequent lines until encountering the same hierarchy type as the first.
    """
    try:
        pattern = r'(<([a-zA-Z]+)[^>]*?>)\((\d+|[a-zA-Z])\)\((\d+|[a-zA-Z])\)(.*)'
        
        def adjust_padding_left(line):
            """
            Add padding-left: 1em to the line's style attribute.
            """
            tag_match = re.match(r'(<[a-zA-Z][^>]*?)>', line)
            if tag_match:
                tag_start = tag_match.group(1)
                if 'style=' in tag_start:
                    updated_tag = re.sub(
                        r'style="([^"]*)"',
                        lambda m: f'style="{m.group(1)}; padding-left: 1em;"',
                        tag_start
                    )
                else:
                    updated_tag = f'{tag_start} style="padding-left: 1em;"'
                return updated_tag + '>' + line[len(tag_match.group(0)):]
            return line

        def classify_hierarchy(hierarchy):
            """
            Determine the regex pattern for the given hierarchy type.
            """
            if re.match(r'^[a-z]$', hierarchy):  # Lowercase letter
                return r'<[a-zA-Z][^>]*?>\([a-z]\)'
            elif re.match(r'^[A-Z]$', hierarchy):  # Uppercase letter
                return r'<[a-zA-Z][^>]*?>\([A-Z]\)'
            elif re.match(r'^\d+$', hierarchy):  # Numeric
                return r'<[a-zA-Z][^>]*?>\(\d+\)'
            elif re.match(r'^[ivx]+$', hierarchy, re.IGNORECASE):  # Roman numeral
                return r'<[a-zA-Z][^>]*?>\([ivxIVX]+\)'
            return None

        lines = html_content.splitlines()
        adjusted_lines = []
        in_double_hierarchy = False
        first_hierarchy_pattern = None


        for line in lines:
            double_hierarchy_match = re.search(pattern, line)
            if double_hierarchy_match:
                # Extract hierarchy components and the rest of the line
                tag_start = double_hierarchy_match.group(1)
                tag_name = double_hierarchy_match.group(2)
                first = double_hierarchy_match.group(3)
                second = double_hierarchy_match.group(4)
                rest_of_line = double_hierarchy_match.group(5).strip()

                # Create split lines
                first_line = f'{tag_start}({first})</{tag_name}>'
                second_line = f'{tag_start}({second}) {rest_of_line}</{tag_name}>'

                adjusted_lines.append(first_line)
                adjusted_lines.append(adjust_padding_left(second_line))

                # Set stopping condition based on the first hierarchy
                first_hierarchy_pattern = classify_hierarchy(first)
                in_double_hierarchy = True
            elif in_double_hierarchy:
                print(f"Line: {line}")
                print(f"Hierarchy: {classify_hierarchy(line)}")
                # Stop adding padding if the same hierarchy type is encountered
                if first_hierarchy_pattern and re.search(first_hierarchy_pattern, line):
                    in_double_hierarchy = False
                    adjusted_lines.append(line)
                elif '<br>' in line:
                    adjusted_lines.append(line)
                else:
                    adjusted_lines.append(adjust_padding_left(line))
            else:
                adjusted_lines.append(line)

        return '\n'.join(adjusted_lines)
    except Exception as e:
        print(f"Error handling double hierarchy: {e}")
        return html_content


def regex_sections(html_content, start_pattern, end_pattern):
    """
    Use regex to extract sections from HTML content and process hierarchical conflicts per section.
    """
    try:
        print("Applying regex...")
        pattern = f'({start_pattern}.*?){end_pattern}'
        raw_matches = re.findall(pattern, html_content, flags=re.DOTALL)
        print(f"Found {len(raw_matches)} section(s).")
        
        processed_matches = []
        for idx, match in enumerate(raw_matches, start=1):
            print(f"Processing Section {idx}...")
            processed_section = handle_hierarchical_conflict(match)
            processed_matches.append(processed_section)
        
        return processed_matches
    except Exception as e:
        print(f"Error during regex extraction: {e}")
        return []
    
def save_html(matched_sections, output_file):
    """
    Save matched sections to a new HTML file.
    """
    try:
        preamble = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Extracted Sections</title>
            <link rel="stylesheet" type="text/css" href="http://127.0.0.1:5500/usc.css" />
        </head>
        <body>
        """
        closing_tags = "</body>\n</html>"

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for idx, section in enumerate(matched_sections, start=1):
                outfile.write(preamble)
                outfile.write(f"<!-- Section {idx} -->\n")
                outfile.write(section)
                outfile.write(closing_tags)

        print(f"Sections saved to {output_file}.")
        return output_file
    except Exception as e:
        print(f"Error saving HTML: {e}")
        return None

def view_html_in_browser(file_path):
    """
    Open the generated HTML file in the default web browser.
    """
    try:
        webbrowser.open(file_path)
        print(f"Opened {file_path} in browser.")
    except Exception as e:
        print(f"Error opening HTML file in browser: {e}")

def main():
    """
    Main function to process HTML.
    """
    # Select .htm file
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("HTM/HTML Files", "*.htm;*.html")])
    if not file_path:
        print("No file selected. Exiting.")
        return

    html_file = convert_htm_to_html(file_path)
    if not html_file:
        print("Error converting file. Exiting.")
        return

    # Read HTML content
    try:
        with open(html_file, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return

    # Get regex patterns from the user
    start_pattern = input("Enter the start regex pattern: ")
    end_pattern = input("Enter the end regex pattern: ")

    # Extract sections using regex
    matched_sections = regex_sections(html_content, start_pattern, end_pattern)
    if not matched_sections:
        print("No matches found. Exiting.")
        return

    # Save extracted sections to a new HTML file
    fileName = input("Enter file name to save: ")
    output_html = fileName + ".html"
    saved_file = save_html(matched_sections, output_html)
    if not saved_file:
        print("Error saving extracted sections. Exiting.")
        return

    # Open the HTML file for user review
    view_html_in_browser(saved_file)

if __name__ == "__main__":
    main()