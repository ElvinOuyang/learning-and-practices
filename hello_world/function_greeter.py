# define a function
def greet_user():  # use "def" to define a function
    """Display a simple greeting."""
    print("Hello!")
greet_user()


# passing information to a function
def greet_user(username):  # "username" is the parameter you need to enter when calling this function
    """Display a simple greeting."""
    print("Hello, " + username.title() + "!")

greet_user('jesse')


# Exercise
# 8.1
def display_message():
    print("I have learnt to define functions and to pass values to functions.")

display_message()


# 8.2
def favorite_book(title):
    print("My favorite book is " + title.title() + ".")

favorite_book("'1984'")


# Passing arguments
def describe_pet(animal_type, pet_name):
    """Display information about a pet."""
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")

describe_pet('hamster','harry')  # positional call, order matters
describe_pet('dog','willie')  # multiple function calls
describe_pet(animal_type='hamster',pet_name='harry')  # keyword arguments, order doesn't matter


# default values
def describe_pet(pet_name, animal_type='dog'):  # default parameters come last
    """Display information about a pet."""
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")

describe_pet(pet_name='willie')
describe_pet('willie')  # the argument would go into the first undefault parameter, which is pet_name


# Exercise
# 8.3
def make_shirt(size, message):
    """Print on the shirt."""
    print(size.upper())
    print(message.title())

make_shirt('s', "say you say me.")
make_shirt(size='l', message="say you say me.")


# 8.4
def make_shirt(size="L", message="I love Python."):
    """Print on the shirt."""
    print(size.upper())
    print(message.title())

make_shirt("m")
make_shirt('l')
make_shirt('s','I love C++')


# 8.5
def describe_city(name, country="China"):
    print(name.title() + " is in " + country.title() + ".")

describe_city("nanchang")
describe_city("beijing")
describe_city('san francisco','united states')


# Returning values
def get_formatted_name(first_name, last_name):
    """Return a full name, neatly formatted."""
    full_name = first_name + ' ' + last_name
    return full_name.title()  # returns the value to the function call

musician = get_formatted_name('jimi','hendrix')
print(musician)


# making an argument optional
def get_formatted_name(first_name, last_name, middle_name=""):  # give optional argument a null value
    """Return a full name, neatly formatted."""
    if middle_name:  # make conditional statements to adjust changes to two situations
        full_name = first_name + ' ' + middle_name + " " + last_name
    else:
        full_name = first_name + ' ' + last_name
    return full_name.title()

musician = get_formatted_name('jimi','hendrix')
print(musician)

musician = get_formatted_name('jimi','hendrix','lee')
print(musician)


# returning a list or a dictionary
def build_person(first_name, last_name, age=''):
    """Return a dictionary of information about a person."""
    person = {'first':first_name,'last':last_name}
    if age:
        person['age'] = age
    return person

musician = build_person('jimi', 'hendrix', 27)
print(musician)


# using functions with while loop
def get_formatted_name(first_name, last_name):
    """Return a full name, neatly formatted."""
    full_name = first_name + ' ' + last_name
    return full_name.title()

while True:
    print("\nPlease tell me your name:")
    print("(enter 'q' at any time to quit)")
    f_name = input("First name: ")
    if f_name == 'q':
        break
    l_name = input("Last name: ")
    if l_name == 'q':
        break

    formatted_name = get_formatted_name(f_name, l_name)
    print("\nHello, " + formatted_name + "!")


# Exercise
# 8.6
def city_country(city_name, country):
    print_string = '"' + city_name.title() + ', ' + country.title() + '"'
    return print_string

print(city_country("nanchang","china"))
print(city_country('beijing','china'))
print(city_country('Washington, DC','united states'))


# 8.7
def make_album(artist, title, track_number=0):
    album = {'artist': artist.title(), 'title': title.title()}
    if track_number:
        album['number of tracks'] = track_number
    return album

print(make_album('jay chou', 'fantasy', 5))
print(make_album('U2', 'unbelievable'))


# 8.8
while True:
    print("Tell me your favorite album:")
    print("(enter 'q' at any time to quit)")
    artist_name = input("Please enter the artist's name: ")
    if artist_name == 'q':
        break
    album_title = input("Please enter album's title: ")
    if album_title == 'q':
        break
    track_number = input("How many tracks in this album? ")
    if track_number == 'q':
        break
    print(make_album(artist_name,album_title,track_number))


# Use functions on a list
def greet_users(names):
    """Print a simple greeting to each user in the list."""
    for name in names:  # Use for loops on a list, which will be defined in function calls
        msg = "Hello, " + name.title() + "!"
        print(msg)

usernames = ['hannah', 'ty', 'margot']
greet_users(usernames)


# Use functions to make codes more efficient
def print_models(unprinted_designs, completed_models):
    """Simulate printing each design, until none are left.
    Move each design to completed_models after printing."""
    while unprinted_designs:
        current_design = unprinted_designs.pop()

        # Simulate creating a 3D print from the design.
        print("Printing model: " + current_design)
        completed_models.append(current_design)

def show_completed_models(completed_models):
    """Show all the models that were printed."""
    print("\nThe following models have been printed:")
    for completed_model in completed_models:
        print(completed_model)

unprinted_designs = ['iphone case', 'robot pendant', 'dodecahedron']
completed_models = []
print_models(unprinted_designs,completed_models)
show_completed_models(completed_models)
