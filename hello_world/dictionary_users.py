user_0 = {
    'username': 'eferm1',
    'first': 'enrico',
    'last': 'fermi',
}

# looping through the dictionary
for key, value in user_0.items():  # .items() is a method that returns a list of key-value pairs
    # ^     ^  these two variables can be defined with any variable name
    print("\nKey: " + key)
    print("Value: " + value)

favorite_language = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
    }

for name, language in favorite_language.items():
    print(name.title() + "'s favorite language is " +
          language.title() + ".")

# looping through all the keys in a dictionary
favorite_language = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
    }

for name in favorite_language.keys():  # keys() returns all the keys in dictionary
    print(name.title())
# above is equal to:
for name in favorite_language:
    print(name.title())

# looping through keys in dictionary with if statement
friends = ['phil','sarah']
for name in favorite_language:  # loop through dictionary
    print(name.title())
    if name in friends:  # if statement, check if within friend list
        print("Hi " + name.title() +
              ", I see your favorite language is " +
              favorite_language[name].title() + "!")

if 'erin' not in favorite_language.keys():
    print("Erin, please take our poll!")

# return items from dictionary in certain order
favorite_language = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
    }

for name in sorted(favorite_language.keys()):  # sort the returned key list in order temporarily
    print(name.title() + ", thank you for taking the poll.")

# looping through all values in a dictionary
favorite_language = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
    }
print("The following languages have been mentioned:")
for language in favorite_language.values():  # values() return all values in a dictionary
    print(language.title())

for language in sorted(set(favorite_language.values())):  # set() a unique list of all elements included
    print(language.title())

"Exercise"
# 6.5
rivers = {
    'nile':'egypt',
    'yellow river': 'china',
    'mississippi': 'usa',
    'rhine': 'france',
    }
for river, country in sorted(rivers.items()):
    print("The " + river.title() + " runs through " +
          country.title() + ".")

river_list = ""
for river in sorted(rivers.keys())[:-1]:
    river_list = river_list + river.title() + ", "
river_list = river_list + sorted(rivers.keys())[-1].title()
print(river_list)

country_list = [country for country in sorted(rivers.values())]
print_list = ""
for country in country_list[:-1]:
    print_list = print_list + country.title() + ", "
print_list = print_list + country_list[-1].upper()
print(print_list)

river_list = [river for river in sorted(rivers.keys())]
country_list = [country for country in sorted(rivers.values())]
print_tasks = [river_list, country_list]
for print_task in print_tasks:
    print_list = ""
    for item in print_task:
        print_list = print_list + item.title() + ", "
    print_list = print_list + print_task[-1].title()
    print(print_list)


# 6.6
favorite_languages = {
    'jen':'python',
    'sarah':'c',
    'edward':'ruby',
    'phil':'python',
    }
target_respondents = ['jen','sarah','jason','neal','kevin']

for individual in sorted(target_respondents):
    if individual in favorite_languages.keys():
        print(individual.title() +
              ", thank you for taking the poll!")
    else:
        print(individual.title() +
              ", please take the poll, thank you!")
