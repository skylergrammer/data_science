#! /usr/bin/evn python

# Standard Libraries   
import argparse

# Non-Standard Libraries
import bs4
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from nltk.corpus import stopwords
import pytagcloud as ptc
from pytagcloud.lang.counter import get_tag_counts


class ParentURLSearch:

    def __init__(self, url, pass_=2):
    
        search_list = [url]
        
        # Counter variables
        nth, npasses = 0, 0
 
        print("Fetching child links for %s..." % url)
        
        # Search for at most 3 passes
        while npasses < pass_:
        
            # Up iteration counter
            npasses += 1
            
            # Reset counter variables
            nold, nnew = 0, 0
            
            print("Beginning pass %s; searching %s links" % (npasses, len(search_list[nth:])))
            for each in search_list[nth:]:
            
                # Get all the links within specified url
                links_within = link_descent(each, url)
                # ID links that are not already in search list
                msk = [links_within.index(x) for x in links_within 
                       if x not in search_list]
                
                # Count up when no new links are found
                if not any(msk):
                    nold += 1
                else:
                    #print("Found %s new links in %s" % (len(msk), each))
                    # Reset counter of no new links when new links are found
                    nold = 0
                    nnew += 1   
                
                # Add thew new links to the search list
                links_within = np.array(links_within)
                search_list += links_within[msk]
            
            # Increase index to start from in search_list
            nth += 1
              
            # Stop searching if old links outnumber new by 3X                
            if nold > 3*nnew and npasses > 1:
                print("No longer finding new links in %s" % url)
                break
            
            # Just as a precaution, take set of search list    
            search_list = list(set(search_list))
                 
        print("Fetching child links for %s complete" % url)
        
        
        self.parent_url = url
        self.links_list = search_list
 
    def get_raw_text(self, stop_words=None):

        raw_text = []

        for url in self.links_list:
            r = requests.get(url, auth=HTTPBasicAuth('user', 'pass'))

            # Find all the paragraphs denoted by the tag <p>
            soup = bs4.BeautifulSoup(r.content).findAll("p")
       
            for paragraph in soup: 
                for section in paragraph.findAll(text=True):
                    
                    if section.split():
                       
                        # Remove punctuation
                        no_punct = " ".join([x.lower() for x in section.split()
                                             if x.isalpha()])
            
                        
                        # Remove stop words
                        no_stop_words = [x.strip(" ") for x in no_punct.split(" ") 
                                         if x not in stop_words]
                    
                        if no_stop_words:
                            # Add words to list
                            try:
                                raw_text += no_stop_words
                            except:
                                continue
            # Join all of the words together as a space-delimited string        
            raw_text = " ".join(raw_text)
    
        return raw_text
     
           
def link_descent(link, parent_url):

        r = requests.get(link, auth=HTTPBasicAuth('user', 'pass'))

        links = bs4.BeautifulSoup(r.content).findAll("a")
        
        links_list = []
        for each in links:
            try:
                link_url = each.get("href")
                split = link_url.split(parent_url)
                
                if len(split) > 1 and any(split):
                    links_list.append(link_url)
            except:
                continue
                
        return  links_list            
 
    
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
    parser.add_argument("--urls", default="http://en.wikipedia.org/wiki/Star", 
                        help="Either a single url or a text file containing a \
                         list of urls.")

    args = parser.parse_args()

    theurls = ParentURLSearch(args.urls)
    
    url_text = theurls.get_raw_text(stop_words=stopwords.words("english"))
    
    print(url_text)
    
    '''
    foo = TextConstruct(raw, except_words)
    foo.generate_cloud()
    '''
        
if __name__ == "__main__":
    main()