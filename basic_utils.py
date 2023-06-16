# import requests
import os
import fitz
from pathlib import Path
import magic
import pypandoc
import uuid


def get_file_name(file):
    # temporary naming convention
    return Path(file).stem

def check_file_type(filename):
    mime = magic.Magic(mime=True)
    file_extension = os.path.splitext(filename)[1]
    mime_type = mime.from_file(filename)
    # Perform actions based on the file type
    if file_extension == '.pdf' or mime_type == 'application/pdf':
        return "pdf"
    elif file_extension == 'txt' or mime_type == 'text/plain':
        return "txt"
    elif file_extension=='.docx' or file_extension=='.odt' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return "doc"
    elif file_extension==".csv":
        return "csv"
    else:
        raise Exception("File type not supported")
    

def convert_to_txt(file):
    file_type = check_file_type(file)
    if (file_type=="pdf"):
        convert_pdf_to_txt(file)
    elif (file_type=="doc"):
        convert_doc_to_txt(file)


def convert_pdf_to_txt(pdf_file):
    pdf = fitz.open(pdf_file)
    text = ""
    file_name=get_file_name(pdf_file)
    for page in pdf:
        text+=page.get_text()
    with open(f'{file_name}.txt', 'a') as f:
        f.write(text)
        f.close()


def convert_doc_to_txt(doc_file):
    file_name=get_file_name(doc_file)
    pypandoc.convert_file(doc_file, 'plain', outputfile=f"{file_name}.txt")



pdf_file = 'resume2023v2.pdf'
odt_file = 'resume2023.odt'
docx_file = 'resume2023.docx'
if __name__ == '__main__':
    check_file_type(docx_file)


