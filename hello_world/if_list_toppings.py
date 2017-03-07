requested_toppings = ['mushrooms','green peppers','extra cheese']

"print each item within a list"
for requested_topping in requested_toppings:
    print("Adding " + requested_topping + ".")

print("\nFinished making your pizza!\n")

"print items based on a simply conditional test"
for requested_topping in requested_toppings:
    if requested_topping == 'green peppers':  # a if statement inside the loop to check the ingredient before printing
        print("Sorry, we are out of green peppers right now.")
    else:
        print("Adding " + requested_topping + ".")

print("\nFinished making your pizza!\n")

"conduct a conditional test to see if a list is empty or not"
requested_toppings = []
if requested_toppings:  # if list_name will return FALSE is the list is empty, jumping directly to the else block
    for requested_topping in requested_toppings:
        print("Adding" + requested_topping + ".")
    print("\nFinished making your pizza!")
else:
    print("Are you sure you want a plain pizza?")

"compare items within two lists"
available_toppings = ['mushrooms','olives','green peppers',
                      'pepperoni','pineapple','extra cheese']  # can also be a tuple if fixed
requested_toppings = ['mushrooms','french fries','extra cheese']

for requested_topping in requested_toppings:
    if requested_topping in available_toppings:  # test if an item is in a list
        print("Adding " + requested_topping + ".")
    else:
        print("Sorry, we don't have " + requested_topping + ".")

print("\nFinished making your pizza!")

"Exercise"
# 5.8
usernames = ['riversome','rosalynnxin','emosrevir','kariri','admin']
for username in usernames:
    if username == "admin":
        print("Hello Admin, would you like to see a status report?")
    else:
        print("Greetings, " + username.title() + ", thank you for logging in again!")

# 5.9
usernames = ['riversome','rosalynnxin','emosrevir','kariri','admin']
if usernames:
    for username in usernames:
        if username == "admin":
            print("Hello Admin, would you like to see a status report?")
        else:
            print("Greetings, " + username.title() + ", thank you for logging in again!")
else:
    print("We need to find some users!")

# 5.10
current_users = ['riversome','rosalynnxin','emosrevir','kariri','admin']
new_users = ['riversome','ririka','wonderguy','rosalynnxin','wondergirl']
for new_user in new_users:
    if new_user.lower() in current_users:
        print("You need to enter another user name!")
    else:
        print("This username: " + new_user +", is available.")

# 5.11
numbers = []  # for loop method to create a new list
for n in range(1,10):
    numbers.append(n)

for number in numbers:
    if number == 1:
        ending = "st"
    elif number == 2:
        ending = "nd"
    elif number == 3:
        ending = "rd"
    elif number > 3:
        ending = "th"
    print(str(number) + ending.lower())

numbers = [n for n in range(1,10)]  # list comprehension method to create a new list
for number in numbers:
    if number == 1:
        ending = "st"
    elif number == 2:
        ending = "nd"
    elif number == 3:
        ending = "rd"
    elif number > 3:
        ending = "th"
    print(str(number) + ending.lower())