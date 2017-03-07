"""if statement"""
age = 19
if age >= 18:  # the simplest form of if statement
    print("You are old enough to vote!")
    print("Have you registered to vote yet?")

"""if else statement"""

age = 19
if age >= 18:
    print("You are old enough to vote!")
    print("Have you registered to vote yet?")
else:  # gives a set of statements when the test fails
    print("Sorry, you are too young to vote.")
    print("Please register to vote as soon as you turn 18!")

"""if-elif-else statement"""

age = 12
if age < 4:
    print("Your admission cost is $0.")
elif age < 18:
    print("Your admission cost is $5.")
else:
    print("Your admission cost is $10.")

# alternative codes (better)
age = 12
if age < 4:
    price = 0
elif age < 18:
    price = 5
elif age < 65:
    price = 10
elif age >= 65:  # since we know the condition for the last block, it's better than else statement
    price = 5
print("Your admission cost is $" + str(price) + ".")  # keep the unchanged parts outside of the loop

# if-elif-else is applicable when conditions are mutually exclusive

"""if statements to test multiple conditions"""
# use a set of if statements to test multiple independent conditions
requested_toppings = ['mushrooms','extra cheese']
if 'mushrooms' in requested_toppings:
    print("Adding mushrooms.")
if 'pepperoni' in requested_toppings:
    print("Adding pepperoni.")
if 'extra cheese' in requested_toppings:
    print("Adding extra cheese.")
print("\nFinished making your pizza!")

"""Exercise"""
# 5.3
alien_color = "green"
if alien_color.lower() == "green":
    print("You just earned 5 points!")
alien_color = "red"
if alien_color.lower() == "green":
    print("You just earned 5 points!")

# 5.4
alien_color = "blue"
if alien_color == "green":
    point = 5
if alien_color != "green":
    point = 10
print("You just earned" + str(point) + " points!")

alien_color = "green"
if alien_color == "green":
    point = 5
if alien_color != "green":
    point = 10
print("You just earned" + str(point) + " points!")

# 5.5
alien_color = "red"
if alien_color.lower() == "green":
    point = 5
if alien_color.lower() == "yellow":
    point = 10
if alien_color.lower() == "red":
    point = 15
print("You just earned" + str(point) + " points!")

# 5.6
age = 70  # mutual exclusive conditions, use if-elif-else statement
if age < 2:
    stage = "baby"
elif age < 13:
    stage = "kid"
elif age < 20:
    stage = "teenager"
elif age < 65:
    stage = "adult"
elif age >= 65:
    stage = "elder"

if stage != "adult" and stage != "elder":  # if else statement to give a/an to different words
    print("You are a " + stage + ".")
else:
    print("You are an " + stage + ".")