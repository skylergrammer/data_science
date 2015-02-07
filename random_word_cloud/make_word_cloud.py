#! /usr/bin/evn python

import bs4
import urllib
import argparse
import pytagcloud as ptc
from pytagcloud.lang.counter import get_tag_counts
import webbrowser

def get_raw_text(url):

    html = urllib.urlopen(url).read().decode("utf8")
    
    soup = bs4.BeautifulSoup(html, "html.parser").prettify()
    
    print(soup.find_all("entry-content"))
    
    return raw

    
class TextConstruct:

    def __init__(self, raw_text, url):        
    
        bad_words = ["rett", "research", "syndrome"]
    
        url_words = url.replace("/",".").replace("-",".").split(".")
        bad_words = bad_words + url_words
        
        filtered = " ".join([x.lower() for x in raw_text.split() 
            if len(x) > 3 and x.isalpha() and x.lower() not in bad_words])
            
        self.filtered = filtered
        tag_counts = get_tag_counts(filtered)
        self.tags = ptc.make_tags(tag_counts, maxsize=100)

    def generate_cloud(self):
    
        ptc.create_tag_image(self.tags, "test.png", size=(1024, 1024), fontname="Droid Sans")
                   
def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    args = parser.parse_args()
    
    raw = get_raw_text(args.url)

    '''
    foo = TextConstruct(raw, args.url)

    foo.generate_cloud() 
    '''   
        
if __name__ == "__main__":
    main()