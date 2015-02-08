#! /usr/bin/evn python

import bs4
import requests
import argparse
import pytagcloud as ptc
from pytagcloud.lang.counter import get_tag_counts
import webbrowser

def get_raw_text(url):

    r = requests.get(url)
    
    soup = bs4.BeautifulSoup(r.content).findAll("p")
    
    raw_text = []
    for paragraph in soup: 
        for section in paragraph.findAll(text=True): 
            good = " ".join([x.lower() for x in section.split() if x.isalnum()])
            raw_text.append(good)
            
    raw_text = " ".join(raw_text)
    
    return raw_text

    
def get_except_words(filename):
    
    with open(filename, "r") as f:
        file_content = f.readlines()
        
        words = [x.strip("\n") for x in file_content]  

    return words
    
    
class TextConstruct:

    def __init__(self, raw_text, url, except_words):        
        
        url_words = url.replace("/",".").replace("-",".").split(".")    
        except_words = except_words + url_words
        
        filtered = " ".join([x for x in raw_text.split() 
                             if len(x) > 2 and x not in except_words])
                             
        tag_counts = get_tag_counts(filtered)
        
        self.filtered = filtered
        self.tags = ptc.make_tags(tag_counts, maxsize=100)
        self.filename = "".join([x for x in url_words])+".png"

    def generate_cloud(self):
    
        ptc.create_tag_image(self.tags, self.filename, size=(1024, 1024), fontname="Droid Sans")
                   
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--e", dest="except_words")
    args = parser.parse_args()
    
    raw = get_raw_text(args.url)
    except_words = get_except_words(args.except_words)

    foo = TextConstruct(raw, args.url, except_words)

    foo.generate_cloud() 
        
if __name__ == "__main__":
    main()