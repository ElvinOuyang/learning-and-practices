magicians = ['alice','david','carolina']
for magician in magicians:  # defining a for loop, repeat the same action for the following section
    print(magician)  # the looped action, meaning each "magician" variable will be printed

magicians = ['alice','david','carolina']
for magician in magicians:  # colon lets Python to interpret the next line as start of loop
    print(magician.title() + ", that was a great trick!")  # indented lines after the for loop means inside the loop
    print("I can't wait for see your next trick, " + magician.title() +".\n")

print("Thank you, everyone. That was a great magic show!\n")  # unindented mark unlooped actions after the for loop

"""
Exercise
"""
  # 4.1
pizzas = ['PizzaHut','Pizza Paradise','Papa Johns','&Pizza']
for pizza in pizzas:
    print(pizza)
print("I really really really love pizzas!\n")

  # 4.2
felines = ['tiger','lion','cat','puma']
for feline in felines:
    print(feline)
    print("A " + feline.lower() + " would make a good show!\n")
print("All these felines are mystical and attractive animals!\n")




for magician in magicians[2:]:
    print(magician.title())
