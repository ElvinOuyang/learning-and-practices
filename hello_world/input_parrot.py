# input() function
message = input("Tell me something, and I will repeat it back to you: ")  # green words are prompt
print(message)  # input() will store whatever input entered by the user in variable "message"

name = input("Plesae enter your name!").lower()  # treat the value stored in input() as a normal variable
print("Hello," + name.title() + "!")

# append strings using += operator
prompt = "If you tell us who you are, we can personalize the messages you see."
prompt += "\nWhat is your first name? "  # operator += will add the additional string at the end

name = input(prompt)
print("\nHello, " + name.title() + "!")

# change data format into numerical with int()
age = input("How old are you?")
age = int(age)  # use int() to change the data format into numerical value
print(age >= 18)

# modulo operator (%)
number = input("Enter a number, and I'll tell you if it's even or odd: ")
number = int(number)

if number % 2 == 0:  # modulo operator returns the remainder when the variable is divided by the value
    print("\nThe number " + str(number) + " is even.")
else:
    print("\nThe number " + str(number) + " is odd.")

"Exercise:"
# 7.1
client_car = input("What kind of car would you like? ").lower()
print("Let me see if I can find you a " + client_car.upper())

# 7.2
seating = int(input("How many people are in your dinner group? "))
if seating > 8:
    print("You will have to wait for a table.")
else:
    print("Your table is ready.")

# 7.3
number = int(input("Please enter a number of your choice: "))
if number % 10 == 0:
    print("\nThe number " + str(number) + " is a multiple of 10.")
else:
    print("\nThe number " + str(number) + " is not a multiple of 10.")

