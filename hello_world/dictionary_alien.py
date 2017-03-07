"Basics of creating and using a dictionary"
alien_0 = {'color':'green','points':5}  # alien_0 stores two values, color and points
# use {} to wrap a dictionary; key-value pairs are included in the form of "key : value"; comma is the delimiter
print(alien_0['color'])  # use dictionary_name[key_name] to extract a certain value stored in dictionary
print(alien_0['points'])

new_points = alien_0['points']
print("You just earned " + str(new_points) + " points!")

"Adding new key-value pairs"
alien_0 = {'color':'green','points':5}
print(alien_0)
alien_0['x_position'] = 0  # directly define new key-value pairs
alien_0['y_position'] = 25
print(alien_0)

"build a dictionary from scratch"
alien_0 = {}
alien_0['color'] = 'green'
alien_0['points'] = 5
print(alien_0)

"Modifying values in a dictionary"
alien_0 = {'color': 'green'}
print("The alien is " + alien_0['color'] + '.')
alien_0['color'] = 'yellow'  # redefine values by directing to the corresponding keys
print("The alien is now " + alien_0['color'] + '.')

alien_0 = {'x_position': 0,'y_position':25,'speed':'medium'}
print("Original x-position: " + str(alien_0['x_position']))

# Move the alien to the right.
# Determine how far to move the alien based on its current speed.
if alien_0['speed'] == 'slow':
    x_increment = 1
elif alien_0['speed'] == 'medium':
    x_increment = 2
elif alien_0['speed'] == 'fast':
    x_increment = 3

# The new position is the old position plus the increment.
alien_0['x_position'] = alien_0['x_position'] + x_increment
print("New x-position: " + str(alien_0['x_position']))

"Deleting a key-value pair"
alien_0 = {'color':'green','points':5}
print(alien_0)
del alien_0['points']  # remove 'points' permanently from alien_0
print(alien_0)

"Dictionary of similar objects"
favorite_languages = {  # press enter and indent when one line is not enough
    'jen':'python',
    'sarah':'c',
    'edward':'ruby',
    'phil':'python',  # good practice to add a comma on the last line
    }
print(favorite_languages)

print("Sarah's favorite language is " +
      favorite_languages['sarah'].title() +
      ".")

"Exercise"
# 6.1
rosalynn_xin = {'first_name':'rosalynn','last_name':'yang','age':'25','city':'washington'}
print(rosalynn_xin['city'].title())

# 6.3
glossary = {
    'tuples':'an immutable list',
    'PEP':'Python Enhancement Proposal',
    'traceback':'a record of where the program went into problem',
    'method':'an action that Python can perform on a piece of data',
    }
print('Tuples ' + ": " + glossary['tuples'].title() + "\n")
print('PEP ' + ": " + glossary['PEP'].title() + "\n")
print('Traceback ' + ": " + glossary['traceback'].title() + "\n")
print('Method ' + ": " + glossary['method'].title() + "\n")