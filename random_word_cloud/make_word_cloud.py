#! /usr/bin/evn python

missing_modules = []
import argparse
import sys
try:
    import bs4
except:
    missing_modules.append("beautifulsoup")
try:
    import requests
except:
    missing_modules.append("requests")
try:
    import pytagcloud as ptc
    from pytagcloud.lang.counter import get_tag_counts
except:
    missing_modules("pytagcloud")

if missing_modules:
    sys.exit("\nYou are missing the following modules: %s\n" 
             % ",".join(missing_modules))


def get_raw_text(url):
    '''
    Take the full source html from specified url and convert that into a single
    string of just the body-of-text content.
    '''

    # Retrieve raw html
    r = requests.get(url)
    
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

    
def get_except_words(filename):
    '''
    Open file containing the words that should not appear in the word cloud.
    '''
    
    with open(filename, "r") as f:
        content = f.readlines()
        
        except_words = [x.strip("\n") for x in content]  

    return except_words
 
    
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
    parser.add_argument("--c", dest="combine", action="store_true", default=False,
                        help="If specified, will assume that a list of urls has \
                        been provided and will make a combined word cloud.")
    parser.add_argument("--e", dest="except_words", default=False, 
                        help="Text document containing words to be excluded.")
    args = parser.parse_args()
    
    # If using exception words, read in the words
    if args.except_words:
        except_words = get_except_words(args.except_words)

    # If using a list of urls, read in the list
    if args.combine:
        url_list = read_url_list(args.url)
        
    else:
        url_list = [args.url]
    
    # Combine
    list_of_raw_strings = []
    for each in url_list:       
        url_raw_text = get_raw_text(each)
        list_of_raw_strings.append(url_raw_text)
    
    raw = " ".join(list_of_raw_strings)

    foo = TextConstruct(raw, except_words)
    foo.generate_cloud() 
        
if __name__ == "__main__":
    main()