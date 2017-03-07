# Getting Started
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

# change the HTML file into a series of objects
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc,'html.parser')  # 'html.parser' defines the parser to use

# Show cleaned Soup with prettify()
print(soup.prettify())

print(soup.title)  # the 'title' tag object
print(soup.title.name)  # the name of the 'title' tag object
print(soup.title.parent.name)  # the name of 'title' tag's parent tag
print(soup.p['class'])  # the value of 'class' attribute for the first 'p' object
print(soup.find_all('a'))  # all 'a' tags
print(soup.find(id="link3"))  # object with 'id' attribute as 'link3'
print(soup.get_text())  # get text from all the tags
for link in soup.find_all('a'):
    print(link.get('href'))  # return value of attribute 'href'
