"""
import requests, bs4
# download the html from the url link
res = requests.get('http://nostarch.com')
res.raise_for_status()  # check the status of returned response
# change the html into bs4 objects
noStarchSoup = bs4.BeautifulSoup(res.text)  # .text returns text without separators
type(noStarchSoup)

import requests, bs4
adb_table = requests.get('http://www.adb.org/projects/search/status/approved?keywords=')
adb_tableSoup = bs4.BeautifulSoup(adb_table.text)
print(adb_tableSoup)
project_links_Soup = adb_tableSoup.select('tbody a')
project_urls = []
while len(project_links_Soup):
    project_url_dict = project_links_Soup.pop()
    project_url = project_url_dict['href']
    project_urls.append(project_url)
print(project_urls)
"""

import requests, bs4
adb_table = requests.get('http://www.adb.org/projects/search/status/approved?keywords=')
adb_tableSoup = bs4.BeautifulSoup(adb_table.content)
adb_tableTag = adb_tableSoup.body.tbody
print(adb_tableTag.prettify())
project_tables_list = adb_tableTag.find_all("span",)
print (type(project_table_list))
