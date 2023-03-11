import tesserocr
import PyPDF2
from PIL import Image

path = "vchat.pdf"
'''
pdf_file = open(path, 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)
num_pages = pdf_reader.numPages
for page_number in range(num_pages):
   page = pdf_reader.getPage(page_number)
   xObject = page['/Resources']['/XObject'].getObject()

   for obj in xObject:
      if xObject[obj]['/Subtype'] == '/Image':
         size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
         data = xObject[obj].getData()
         mode = ""
         if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
            mode = "RGB"
         else:
            mode = "P"
         image = Image.frombytes(mode, size, data)
         image.save(obj[1:] + ".png")
   text = tesserocr.image_to_string(Image.open('image.png'))
   print(text)
'''
# Open the PDF file in binary mode
with open(path, 'rb') as pdf_file:
    # Create a PdfReader object using the binary stream of the PDF file
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Get the total number of pages in the PDF file
    num_pages = len(pdf_reader.pages)
    # Extract text from the page using PyPDF2's built-in methods
    for page_number in range(num_pages):
       page = pdf_reader.pages[page_number]  # Subtract 1 to match 0-based index
       text = page.extract_text()
       print(text)