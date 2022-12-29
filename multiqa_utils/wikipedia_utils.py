import jsonlines
import html
import re

def load_postprocess_dump(infile, outfile):
    clean = re.compile('<.*?>')

    postprocess_pages = []
    with jsonlines.open(infile) as reader:
        for obj in reader:
            if obj['text']:
                cleaned_text = re.sub(clean, '', html.unescape(obj['text']))
                new_text = f"Title: {obj['title']}\nArticle: {cleaned_text}"
                postprocess_pages.append({
                    "id": obj['id'],
                    "contents": new_text,
                })

    with jsonlines.open(outfile, mode='w') as writer:
        writer.write_all(postprocess_pages)
    print(f">> Wrote: {outfile}")
