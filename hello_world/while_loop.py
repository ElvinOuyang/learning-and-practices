"while loop"
current_number = 1
while current_number <= 5:
    print(current_number)
    current_number += 1  # short version of current_number = current_number + 1

'while loop with input() function'
prompt = "\nTell me something, and I will repeat it back to you: "
prompt += "\nEnter 'quit' to end the program."
message = ""  # if the variable is not defined, Python won't be able to carry out the while loop
while message != 'quit':  # if message == 'quit', Python would not do anything
    message = input(prompt)
    if message != 'quit':
        print(message)

'flag variable'
prompt = "\nTell me something, and I will repeat it back to you: "
prompt += "\nEnter 'quit' to end the program."

active = True  # flag is set to True
while active:  # as long as variable "active" is True, the loop keeps going
    message = input(prompt)

    if message == 'quit':
        active = False
    else:
        print(message)

'break statement'
prompt = "\nPlease enter the name of a city you have visited:"
prompt += "\n(Enter 'quit' when you are finished.)"

while True:
    city = input(prompt)

    if city == 'quit':
        break  # Python junmps out to the loop immediately
    else:
        print("I'd love to go to " + city.title() + "!")

'continue statement'
current_number = 0
while current_number < 10:
    current_number += 1
    if current_number % 2 == 0:
        continue  # the loop skips even numbers before it directly jumps to next round of loop
    print(current_number)

# 7.4
prompt = "\nPlease enter the topping you would love to add: "
prompt += "\n(Enter 'quit' when you are finished.)"

while True:
    topping = input(prompt)
    if topping == 'quit':
        break
    else:
        print("\nYou will add " + topping.title() + " to the pizza.")

# 7.5:
prompt = "\nPlease enter the topping you would love to add: "
prompt += "\n(Enter 'quit' when you are finished.)"

topping = ""
active = True

while active:
    topping = input(prompt)
    if topping == 'quit':
        active = False
    else:
        print("\nYou will add " + topping.title() + " to the pizza.")

'manipulation of list with while loop'
unconfirmed_users = ['alice','brian','candace']
confirmed_users = []

while unconfirmed_users:
    current_user = unconfirmed_users.pop()  # pop() removes items from the list one at a time from the end
# .pop() stores the item that is "popped" from the list
    print("Verifying user: " + current_user.title())
    confirmed_users.append(current_user)
print("\nThe following users have been confirmed:")
for confirmed_user in confirmed_users:
    print(confirmed_user.title())

'remove all instances of specific values from a list'
pets = ['dog','cat','dog','goldfish','cat','rabbit','cat']
print(pets)

while 'cat' in pets:  # conditional testing until all 'cat's are removed
    pets.remove('cat')
print(pets)

'filling a dictionary with user input'
responses = {}

polling_active = True  # set a flag to indicate that the polling is active
while polling_active:
    name = input("\nWhat is your name? ")
    response = input("Which mountain would you like to climb someday? ")

    responses[name] = response  # store the response in the dictionary with name as key

    repeat = input("Would you like to let another person respond? (yes/no) ")
    if repeat == 'no':
        polling_active = False

print("\n--- Poll Results ---")
for name, response in sorted(responses.items()):
    print(name.title() + " would like to climb " + response.title() + ".")

'Exercise'
# 7.8
sandwich_orders = ['shack burger','smokehouse','wreck','chicken sandwich']
finished_sandwiches = []

while sandwich_orders:
    sandwich_order = sandwich_orders.pop()
    print("You have ordered " + sandwich_order.title() + ".")
    finished_sandwiches.append(sandwich_order)

for finished_sandwich in finished_sandwiches:
    print("We have made " + finished_sandwich.title() + " for you.")

# 7.9
sandwich_orders = ['pastrami','shack burger','smokehouse','pastrami','pastrami','wreck','chicken sandwich']
finished_sandwiches = []
print("The Deli has run out of pastrami.")
while 'pastrami' in sandwich_orders: # remove 'pastrami' out of all orders
    sandwich_orders.remove('pastrami')
print(sandwich_orders)

while sandwich_orders:
    sandwich_order = sandwich_orders.pop()
    print("You have ordered " + sandwich_order.title() + ".")
    finished_sandwiches.append(sandwich_order)

for finished_sandwich in finished_sandwiches:
    print("We have made " + finished_sandwich.title() + " for you.")

# 7.10
prompt = "If you could visit one place in the world,"
prompt += "\n where would you go? "

dream_vacations = {}
polling_active = True
while polling_active:
# record the responses of the respondents first
    name = input("What is your name?\n")
    destination = input(prompt)
    dream_vacations[name] = destination

# decide if the poll is to continue later
    repeat = input("Would you like others to continue the poll? yes/no").lower()
    if repeat == 'no':
        polling_active = False

for name, destination in dream_vacations.items():
    print(name.title() + " would go to " + destination.title() + ".")
