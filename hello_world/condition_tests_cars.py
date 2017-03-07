"test of equality"

cars = ['audi','bmw','subaru','toyota']
for car in cars:
    if car == 'bmw':  # double equality sign "==" is used for the testing
        print(car.upper())  # conditional tests to determine if a variable meets the condition
    else:
        print(car.title())

"test of inequality"
requested_topping = 'mushroom'
if requested_topping != 'anchovies':
    print("Hold the anchovies!")

answer = 17
if answer != 42:  # != tests inequality conditions
    print('That is not the correct answer. Please try again!')

"test if an item is in a list"
requested_toppings = ['mushrooms','onions','pineapple']
print('mushrooms' in requested_toppings)  # keyword "in" here tests if a value appears in a list
print('pepperoni' in requested_toppings)

"test if an item is not in a list"
banned_users = ['andrew','carolina','david']
user = 'marie'
if user not in banned_users:  # test if a value is not in a list, return "TRUE" if not included in the list
    print(user.title() + ", you can post a response if you wish.")
