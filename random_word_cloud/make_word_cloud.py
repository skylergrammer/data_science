#! /usr/bin/evn python

missing_modules = []
import argparse
import bs4
import numpy as np
import requests
from nltk.corpus import stopwords
import pytagcloud as ptc
from pytagcloud.lang.counter import get_tag_counts

class ParentURLs:

    def __init__(self, url):
    
        self.parent_url = url

        search_list = [url]
        
        nth, npasses = 0, 0
 
        print("Fetching child links for %s..." % url)
        while npasses < 3:
            npasses += 1
            nold, nnew = 0, 0
            print("Beginning pass %s; searching %s links" % (npasses, len(search_list[nth:])))
            for each in search_list[nth:]:
                links_within = link_descent(each, url)

                msk = [links_within.index(x) for x in links_within 
                       if x not in search_list]
                
                if not any(msk):
                    nold += 1
                else:
                    print("Found %s new links in %s" % (len(msk), each))
                    nold = 0
                    nnew += 1   
                
                if nold >= 5*nnew and nold > 10:
                    break
                
                links_within = np.array(links_within)
                search_list += links_within[msk]
            nth += 1
                            
            if nold > nnew:
                print("No longer finding new links...")
                break
                
            search_list = list(set(search_list))
                 
        print("Fetching child links for %s complete" % url)
        
           
def link_descent(link, parent_url):

        r = requests.get(link)
    
        links = bs4.BeautifulSoup(r.content).findAll("a")

        links_list = []
        for each in links:
            try:
                link_url = each.get("href")
                ext = link_url.split(".")[-1]
                split = link_url.split(parent_url)
                if len(split) > 1 and ext == "html" and any(split):
                    links_list.append(link_url)
            except:
                continue
        return  links_list            

def get_raw_text(r):
    '''
    Take the full source html from specified url and convert that into a single
    string of just the body-of-text content.
    '''

    # Find all the paragraphs denoted by the tag <p>
    soup = bs4.BeautifulSoup(r.content).findAll("p")
   
    # Iterate over paragraphs to remove non-alphanumeric characters
    raw_text = []
    for paragraph in soup: 
        for section in paragraph.findAll(text=True): 
            good = " ".join([x.lower() for x in section.split() if x.isalnum()])
            raw_text.append(good)
     
    # Join all of the characters together as a space-delimited string        
    raw_text = " ".join(raw_text)
    
    return raw_text
 
    
def read_url_list(filename):
    '''
    Read file with urls.  Returns a list of urls.
    '''
    
    with open(filename, "r") as f:
        content = f.readlines()
        
        urls = [x.strip("\n") for x in content]
          
    return urls
   
    
class TextConstruct:

    def __init__(self, raw_text, except_words):        
                
        # Remove words shorter than 2 chars and words in except list
        filtered = " ".join([x for x in raw_text.split() 
                             if len(x) > 2 and x not in except_words])
         
        # Get word counts for each word in filtered text                     
        tag_counts = get_tag_counts(filtered)
        
        self.filtered = filtered
        self.tags = ptc.make_tags(tag_counts, maxsize=60, minsize=6)

    def generate_cloud(self):
    
        ptc.create_tag_image(self.tags, "word_cloud.png", 
                             size=(1024, 1024), fontname="Droid Sans", 
                             rectangular=False)
                   
def main():

    parser = argparse.ArgumentParser(description="Will create a word cloud of \
                                     the content in url(s).  NOTE: pytagcloud \
                                     can be very slow.  Be patient.")
    parser.add_argument("--url", default="http://en.wikipedia.org/wiki/Star", 
                        help="Either a single url or a text file containing a \
                         list of urls.")

    args = parser.parse_args()
    

   
    theurl = ParentURLs(args.url)
    
    '''
    foo = TextConstruct(raw, except_words)
    foo.generate_cloud()
    '''
        
if __name__ == "__main__":
    main()