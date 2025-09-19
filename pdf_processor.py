#pdf_processor.py
import fitz  # PyMuPDF
import os

def extract_text_by_page(pdf_path):
    """
    Extracts text per page from the PDF and returns a list of tuples:
    [(page_number (1-based), page_text), ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        page_text = page.get_text("text")
        page_text = page_text.replace("\r", "\n")
        pages.append((i + 1, page_text))
    doc.close()
    return pages

def extract_images(pdf_path, out_dir="static/images"):
    """
    Extract images from each page of the PDF.
    Returns metadata list: [{"id", "page", "path"}, ...]
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images_meta = []
    img_id = 0
    for page_num, page in enumerate(doc, start=1):
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 4:
                img_data = pix.tobytes("png")
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix.tobytes("png")
            img_path = os.path.join(out_dir, f"page{page_num}_img{i}.png")
            with open(img_path, "wb") as f:
                f.write(img_data)
            images_meta.append({
                "id": f"img_{img_id}",
                "page": page_num,
                "path": img_path
            })
            img_id += 1
            pix = None
    doc.close()
    return images_meta

if __name__ == "__main__":
    pdf_file = "AmpD Enertainer User Manual (NCM) - Rev 2.3.pdf"
    pages = extract_text_by_page(pdf_file)
    print(f"Extracted {len(pages)} pages.")
    images = extract_images(pdf_file)
    print(f"Extracted {len(images)} images.")
