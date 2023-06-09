# import requests
import os
import fitz
from pathlib import Path
import magic
import pypandoc
import uuid
import markdown
import csv
from openai_api import get_moderation_flag, check_injection



# def check_file_type(filename):
#     mime = magic.Magic(mime=True)
#     file_extension = os.path.splitext(filename)[1]
#     mime_type = mime.from_file(filename)
#     # Perform actions based on the file type
#     if file_extension == '.pdf' or mime_type == 'application/pdf':
#         return "pdf"
#     elif file_extension == 'txt' or mime_type == 'text/plain':
#         return "txt"
#     elif file_extension=='.docx' or file_extension=='.odt' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
#         return "doc"
#     elif file_extension==".csv":
#         return "csv"
#     else:
#         raise Exception("File type not supported")
    

def convert_to_txt(file, output_path):
    file_ext = os.path.splitext(file)[1]
    if (file_ext=='.pdf'):
        try: 
            convert_pdf_to_txt(file, output_path)
        except Exception as e:
            print(e)
    elif (file_ext=='.odt' or file_ext=='.docx'):
        try:
            convert_doc_to_txt(file, output_path)
        except Exception as e:
            print(e)


def convert_pdf_to_txt(pdf_file, output_path):
    pdf = fitz.open(pdf_file)
    text = ""
    for page in pdf:
        text+=page.get_text()
    with open(output_path, 'a') as f:
        f.write(text)
        f.close()


def convert_doc_to_txt(doc_file, output_path):
    pypandoc.convert_file(doc_file, 'plain', outputfile=output_path)

def read_txt(file):
    try:
        with open(file, 'r', errors='ignore') as f:
            text = f.read()
            return text
    except Exception as e:
        print(e)
    
def markdown_table_to_dict(markdown_table):
    # Convert Markdown to HTML
    html = markdown.markdown(markdown_table)
    print(html)

    # Parse HTML table using csv module
    rows = csv.reader(html.split('\n'), delimiter='|')

    # Extract header and data rows
    header = next(rows)
    data = [row for row in rows if len(row) > 1]

    # Convert rows to dictionary
    result = []
    for row in data:
        item = {}
        for i, value in enumerate(row):
            key = header[i].strip()
            item[key] = value.strip()
        result.append(item)

    return result


def check_content_safety(file=None, text_str=None):
    if (file!=None):
        text = read_txt(file)
    elif (text_str!=None):
        text = text_str
    if (get_moderation_flag(text) or check_injection(text)):
        return False
    else:
        return True



