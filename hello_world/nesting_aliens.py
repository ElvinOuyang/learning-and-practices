"Nesting"
# Nesting dictionaries under list
alien_0 = {'color': 'green','points': 5}
alien_1 = {'color': 'yellow','points': 10}
alien_2 = {'color': 'red','points': 15}
aliens = [alien_0, alien_1, alien_2]

for alien in aliens:
    print(alien)

# Looping to create a list of dictionaries
aliens = []  # make an empty to store aliens

for alien_number in range(30):  # make 30 green aliens
    new_alien = {'color': 'green','points': 5, 'speed': 'slow'}
    aliens.append(new_alien)
for alien in aliens[0:3]:  # modify the aliens list
    if alien['color'] == 'green':
        alien['color'] = 'yellow'
        alien['speed'] = 'medium'
        alien['points'] = 10
    elif alien['color'] == 'yellow':
        alien['color'] = 'red'
        alien['speed'] = 'fast'
        alien['points'] = 15

for alien in aliens[:5]:  # show the first 5 aliens
    print(alien)
print("...")
print("Total number of aliens: " + str(len(aliens)))  # print total number of aliens

# Nesting a list in a dictionary
pizza = {  # store information about a pizza being ordered
    'crust': 'thick',  # two different key-value pairs
    'toppings': ['mushrooms', 'extra cheese'],
    }

print("You ordered a " + pizza['crust'] + '-crust pizza ' +
      "with the following toppings:")
for topping in pizza['toppings']:
    print("\t" + topping)