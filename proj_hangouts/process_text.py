#! /usr/bin/env python
import re
from collections import Counter


def get_wordcount(text,ignorelist = None):
    text = text.lower()
    words = re.findall(r"\w[\w']*",text)

    wordcounter = Counter(words)

    if ignorelist:
        for word in ignorelist:
            del wordcounter[word]

    for word in wordcounter.keys():
        if word.isdigit():
            del wordcounter[word]


    print wordcounter.most_common(10)


textfile = open('logfile','r')
text = textfile.readlines()
text = ' '.join(text)
textfile.close()

ignorefile = open('ignorelist','r')
ignorelist = ignorefile.readlines()
ignorelist = [x.strip() for x in ignorelist]
ignorefile.close()

ignorelist += ['michael','skyler','pyc','hangouts','gordon','grammer']
get_wordcount(text,ignorelist)
