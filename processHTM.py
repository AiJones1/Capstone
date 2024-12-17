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
    Handle lines with multiple hierarchical elements.
    Splits such lines and adjusts padding-left dynamically for the second hierarchy.
    Also loops over subsequent lines to find the next instance of the same hierarchy type and prints them.
    """
    try:
        pattern = r'(<([a-zA-Z]+)[^>]*?>)\((\d+|[a-zA-Z])\)\((\d+|[a-zA-Z])\)'
        
        css_pattern = r'<style.*?>(.*?)</style>'
        css_matches = re.findall(css_pattern, html_content, flags=re.DOTALL)
        embedded_css = css_matches[0] if css_matches else ""

        def get_adjusted_padding(element_class):
            css_rule_pattern = rf'\.{element_class}\s*\{{.*?padding-left:\s*([\d.]+em);.*?\}}'
            match = re.search(css_rule_pattern, embedded_css)
            if match:
                current_padding = float(match.group(1).replace('em', ''))
                return current_padding + 1
            return 1
        
        def adjust_padding_left(line):
            """
            Adjust padding-left of an HTML tag in the given line.
            """
            try:
                # Match the first opening tag and locate the closing bracket
                tag_match = re.match(r'(<[a-zA-Z][^>]*?)>', line)
                if tag_match:
                    tag_start = tag_match.group(1)

                    # Check if a style attribute already exists
                    if 'style=' in tag_start:
                        # Append the padding to the existing style attribute
                        updated_tag = re.sub(
                            r'style="([^"]*)"',
                            lambda m: f'style="{m.group(1)}; padding-left: 1em;"',
                            tag_start
                        )
                    else:
                        updated_tag = f'{tag_start} style="padding-left: 1em;"'

                    # Replace the original tag in the line with the updated tag
                    return updated_tag + '>' + line[len(tag_match.group(0)):]
                return line
            except Exception as e:
                print(f"Error adjusting padding: {e}")
                return line
            
        def classify_hierarchy(hierarchy):
            """
            Classify the hierarchy type and return the corresponding regex pattern for matching.
            """
            if re.match(r'^[a-z]$', hierarchy):
                return r'^<.*?>\([a-z]\)'  
            elif re.match(r'^[A-Z]$', hierarchy):
                return r'^<.*?>\([A-Z]\)'  
            elif re.match(r'^\d+$', hierarchy):
                return r'^<.*?>\(\d+\)' 
            elif re.match(r'^[ivx]+$', hierarchy, re.IGNORECASE):
                return r'^<.*?>\([ivx]+\)' 
            return None

        def replacer(match):
            tag_start = match.group(1)
            tag_name = match.group(2)
            first = match.group(3)
            second = match.group(4)

            class_match = re.search(r'class="([^"]+)"', tag_start)
            element_class = class_match.group(1) if class_match else ""
            adjusted_padding = get_adjusted_padding(element_class)

            updated_tag_second = f'<{tag_name} class="{element_class}" style="padding-left: {adjusted_padding}em;">'
            
            first_type = classify_hierarchy(first)
            print(f"First type: {first_type}")

            lines = html_content.splitlines()
            adjusted_lines=[]
            found_start = False

            for i, line in enumerate(lines):
              
                if not found_start:
                    
                    double_hierarchy_match = re.search(r'\(([a-zA-Z0-9]+)\)\(([a-zA-Z0-9]+)\)', line)
                    if double_hierarchy_match:
                        print(f"Found double hierarchy: {line}")

                        first_hierarchy = double_hierarchy_match.group(1)
                        if not first_hierarchy:
                            print(f"Could not extract first hierarchy from line: {line}")
                            break

                        first_hierarchy_regex = classify_hierarchy(first_hierarchy)
                        if not first_hierarchy_regex:
                            print(f"Unknown hierarchy type for: {first_hierarchy}")
                            break
                    else:
                        adjusted_lines.append(line)

                        found_start = True 
                elif found_start:
                    if re.match(first_hierarchy_regex, line):
                        print(f"Stopping at line with same hierarchy type: {line}")
                        break
                    
                    print(f"Processing line: {lines[i]}")
                    adjusted = adjust_padding_left(line)
                    changed_line =replace_line_directly(html_content, line, adjusted)
                    print(f"changed line: {changed_line}")
                    adjusted_lines.append(adjusted)

            new_content = (
                f'{tag_start}({first})</{tag_name}>\n'
                f'{updated_tag_second}({second})'
            )
            return new_content + '\n'.join(adjusted_lines)
    except Exception as e:
        print(f"Error handling hierarchical conflict: {e}")
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
            # IF COMES ACCROSS USE THE OTHER FUNCTION to do 
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
