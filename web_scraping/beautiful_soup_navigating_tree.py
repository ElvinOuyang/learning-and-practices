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

soup = BeautifulSoup(html_doc, 'lxml')

'Going Down'
'Navigating using tag names'
print(soup.head)
# <head><title>The Dormouse's story</title></head>
print(soup.title)
# <title>The Dormouse's story</title>
print(soup.body.b)
# <b>The Dormouse's story</b>

# Note: using the tag name only gives you the first tag by that name

'.contents and .children'
head_tag = soup.head
print(head_tag)
# <head><title>The Dormouse's story</title></head>
print(head_tag.contents)
# [<title>The Dormouse's story</title>]
# .contents returns the objects as a list
title_tag = head_tag.contents[0]
print(title_tag)
# <title>The Dormouse's story</title>
# Use list reference [n] to locate the specific string
print(title_tag.contents)
# ["The Dormouse's story"]
# returns the string as a list

for child in title_tag.children:
    print(child)
# The Dormouse's story -- NavigableString inside of title_tag
# <strong> Good</strong> -- 'strong' Tag inside of title_tag
# use .children to extract the tag's DIRECT children in an iteration

'.descendants'
for child in head_tag.descendants:
    print(child)
# <title>The Dormouse's story</title> -- the 'title' tag inside of 'head' tag
# The Dormouse's story -- the NavigableString inside of 'title' tag
# .descendant will find all children at all level of the Tag

'.string'
print(title_tag.string + "1")
# The Dormouse's story1
# .string is only available when the NavigableString is the only child of a Tag

print(soup.html.string)
# None
# When the Tag contains more than one child, .string will only return None

'.strings and stripped_strings'
for string in soup.strings:
    print(repr(string))
# .strings return a list of all the strings inside a Tag

for string in soup.stripped_strings:
    print(repr(string))
# .stripped_strings return a list of non-white-space strings inside a Tag

'Going Up'
'.parent'
title_tag = soup.title
print(title_tag)
# <title>The Dormouse's story</title>
print(title_tag.parent)
# <head><title>The Dormouse's story</title></head>

print(title_tag.string.parent)
# <title>The Dormouse's story</title>
# the title string (a NavigableString) itself has a parent, which is the title tag

'.parents'
link = soup.a
print(link)
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
for parent in link.parents:
    if parent is None:
        print(parent)
    else:
        print(parent.name)
# .parents will return a list of parents of the Tag

'Going sideways'
sibling_soup = BeautifulSoup("<a><b>text1</b><c>text2</c></b></a>",'lxml')
print(sibling_soup.prettify())
# <html>
#  <body>
#   <a>
#    <b>
#     text1
#    </b>
#    <c>
#     text2
#    </c>
#   </a>
#  </body>
# </html>

# siblings: tags at the same level
'.next_sibling and .previous_sibling'
print(sibling_soup.b.next_sibling)
# <c>text2</c>
# returns the next tag at the same level

print(sibling_soup.c.previous_sibling)
# <b>text1</b>
# returns the previous tag at the same level

print(sibling_soup.b.previous_sibling)
# None
# When no same level tag exists, Python will return "None"

link = soup.a
print(link)
print(link.next_sibling)
print(link.next_sibling.next_sibling)
# In most cases, the next sibling of a Tag is a while space or a comma

'.next_siblings and .previous_siblings'
for sibling in soup.a.next_siblings:
    if sibling.name == 'a':
        print(repr(sibling))

for sibling in soup.find(id='link3').previous_siblings:
    if sibling.name == 'a':
        print(repr(sibling))

'Going back and forth'
'.next_element and .previous_element'
last_a_tag = soup.find(id="link3")
print(last_a_tag)
print(last_a_tag.next_sibling)
# and they lived at the bottom of a well.
print(last_a_tag.next_element)
# Tillie

'.next_elements and .previous_elements'
for element in last_a_tag.next_elements:
    print(repr(element))
