from bs4 import BeautifulSoup
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

soup = BeautifulSoup(html_doc, 'html.parser')

'Search Filter Type'
'A string'
print(soup.find_all('b'))
# [<b>The Dormouse's story</b>]
# returns a list that includes all 'b' Tags

'A regular expression'
import re
for tag in soup.find_all(re.compile('^b')):
    print(tag.name)
# body
# b
# re.compile() creates a regular expression. "^" indicates start of the string

for tag in soup.find_all(re.compile('t')):
    print(tag.name)

'A list'
print(soup.find_all(['a','b']))
# [<b>The Dormouse's story</b>,
#  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

# Note: If searching a list, Beautiful Soup will allow string match against any item in that list

'True'
for tag in soup.find_all(True):
    print(tag.name)
# find by True will return all the Tags but no NavigableString

'A function'
def has_class_but_no_id(tag):
    # A logical test that returns True if the tag has attribute 'class' but no attribute 'id'
    return tag.has_attr('class') and not tag.has_attr('id')
# Note: the argument 'tag' makes sure that the function will only test the tags within a soup

print(soup.find_all(has_class_but_no_id))
# [<p class="title"><b>The Dormouse's story</b></p>,
#  <p class="story">Once upon a time there were...</p>,
#  <p class="story">...</p>]
# find_all using a function will return all tag that meet the requirements

def not_lacie(href):
    return href and not re.compile("lacie").search(href)
print(soup.find_all(href=not_lacie))
# a function that finds all a tags with attribute href not matching a regular expression

'The name argument'
print(soup.find_all("title"))
# [<title>The Dormouse's story</title>]
# Only return tags with the specific names

'The keyword argument'
print(soup.find_all(id='link2'))
# [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]
# An extract match for attribute 'id'

print(soup.find_all(id=True))
# A search for all Tags with an 'id' attribute

print(soup.find_all(href=re.compile("elsie")))
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]
# Matching just part of the attribute 'href'

# Keyword arguments can be listed together to narrow down the search

# KEY: if encountering SyntaxError
data_soup = BeautifulSoup('<div data-foo="value">foo!</div>',"lxml")
print(data_soup.find_all(attrs={"data-foo":"value"}))
# Turn the "attribute_name=search_value" into "attrs={attribute_name:search_value)" will fix the problem

'Searching by CSS class'
# Use class_ as keyword argument when searching for CSS class
print(soup.find_all("a",class_="sister"))

# What about a single CSS class that has more than one value?
css_soup = BeautifulSoup('<p class="body strikeout"></p>','lxml')
css_soup.find_all("p", class_="strikeout")
# [<p class="body strikeout"></p>]
css_soup.find_all("p", class_="body")
# [<p class="body strikeout"></p>]
# Python will return a tag when whichever one of the value matches

# What if I want to match two or more CSS class value in one search?
print(css_soup.select("p.strikeout.body"))
# Use CSS selector

'The string argument'
# search for strings instead of tags
print(soup.find_all(string="Elsie"))
# Returns the exact string

print(soup.find('a', string=re.compile("Elsie")))
# Search for 'a' tags that has string matching Elsie

'recursive argument'
# if recursive=False, only search for direct children


