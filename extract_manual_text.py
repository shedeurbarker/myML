import pdfplumber

pdf_path = 'Manual.pdf'
out_path = 'manual_extracted.txt'

with pdfplumber.open(pdf_path) as pdf:
    all_text = ''
    for page in pdf.pages:
        all_text += page.extract_text() or ''
        all_text += '\n\n'

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(all_text)

print(f'Extracted text saved to {out_path}') 