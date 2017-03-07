from bs4 import BeautifulSoup

'''Beautiful Soup Step by Step'''
'Objects in Beautiful Soup'
'Tag'
soup = BeautifulSoup('<b class="boldest">Extremely bold</b>','lxml')
tag = soup.b
print(type(tag))
# <class 'bs4.element.Tag'>
# most important features of a tag are name and attributes

'Name'
print(tag.name)
# u 'b'
# name of a tag. accessible as .name
# one can also change the name of a tag

'Attributes'
print(tag['class'])
# ['boldest']
# one can access the tag value by treating the tag as a dictionary

print(tag.attrs)
# {'class': ['boldest']}
# .attrs returns attributes directly as dictionaries

'Multi-valued attributes'
css_soup = BeautifulSoup('<p class="body strikeout"></p>', 'lxml')
print(css_soup.p['class'])
# ["body", "strikeout"]
# The multiple values of 'class' is returned as a list

id_soup = BeautifulSoup('<p id="my id"></p>', 'lxml')
print(id_soup.p['id'])
# 'my id'
# in this case, the 'id' attribute is just a single-variable attribute

css_soup = BeautifulSoup('<p class="body strikeout"></p>', 'xml')
print(css_soup.p['class'])
# 'body strikeout'
# if parsed with XML, there would be no multi-valued attributes

'NavigableString'
soup = BeautifulSoup('<b class="boldest">Extremely bold</b>','lxml')
tag = soup.b
print(tag.string)
# 'Extremely bold'
# .string extract the NavigableString from the Tag

print(str(tag.string))
# always reformat the NavigableString into unicode with str()
# NavigableString doesn't support .contents, .string attributes or .find() method

'Beautiful Soup'
print(soup.name)
# the BeautifulSoup object can be treated as a Tag object with no name and no attributes


