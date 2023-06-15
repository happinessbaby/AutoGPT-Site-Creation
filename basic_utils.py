# import requests
import os
import fitz
from pathlib import Path
import magic
import pypandoc

def check_file_type(filename):
    mime = magic.Magic(mime=True)
    file_extension = os.path.splitext(filename)[1]
    mime_type = mime.from_file(filename)
    # Perform actions based on the file type
    if file_extension == '.pdf' or mime_type == 'application/pdf':
        # File is a PDF, perform PDF-specific actions
        convert_pdf_to_txt(filename)
        return "pdf"
    elif file_extension == 'txt' or mime_type == 'text/plain':
        # File is a TXT, perform TXT-specific actions
        return "txt"
    elif file_extension=='.docx' or file_extension=='.odt' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        convert_doc_to_txt(filename)
        return "doc"
    elif file_extension==".csv":
        return "csv"
    else:
        print("file type not supported")


def convert_pdf_to_txt(pdf_file):
    pdf = fitz.open(pdf_file)
    text = ""
    file_name=Path(pdf_file).stem
    for page in pdf:
        text+=page.get_text()
        # print(text)
    with open(f'{file_name}.txt', 'a') as f:
        f.write(text)
        f.close()


def convert_doc_to_txt(doc_file):
    file_name=Path(doc_file).stem
    pypandoc.convert_file(doc_file, 'plain', outputfile=f"{file_name}.txt")



pdf_file = 'resume2023v2.pdf'
odt_file = 'resume2023.odt'
docx_file = 'resume2023.docx'
if __name__ == '__main__':
    check_file_type(docx_file)


