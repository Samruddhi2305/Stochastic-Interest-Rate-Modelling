import pypdf

def extract_urls(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    urls = []
    
    # 1. Look in page annotations for URI actions
    for page_num, page in enumerate(reader.pages):
        if '/Annots' in page:
            annotations = page['/Annots']
            for annot in annotations:
                annot_obj = annot.get_object()
                if '/A' in annot_obj:
                    action = annot_obj['/A'].get_object()
                    if '/URI' in action:
                        urls.append((f"Page {page_num+1} link", action['/URI']))
                        
    # 2. Also extract text and search for text URLs
    import re
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        text_urls = re.findall(r'https?://[^\s()<>"]+', text)
        for url in text_urls:
            urls.append((f"Page {page_num+1} text", url))
            
    return urls

if __name__ == '__main__':
    found_urls = extract_urls('5_6127588626697036839.pdf')
    print("Found URLs:")
    for source, url in found_urls:
        print(f"[{source}]: {url}")
