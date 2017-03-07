dimensions = (200,50)
print(dimensions[0])
print(dimensions[1])
for dimension in dimensions:
    print(dimension)

dimensions = (200,50)
print("Original dimensions:")
for dimension in dimensions:
    print(dimension)

dimensions = (400,100)  # redefine dimensions is allowed
print("\nModified dimensions:")
for dimension in dimensions:
    print(dimension)

"""
Exercise
"""

# 4.13
buffets = ('sausage','cheese','omelet','roast beef','dumpling')
message = "Today's buffet has "
for buffet in buffets[:-1]:  # defining a loop for everything before the last item
    message = message + buffet.title() +", "
message = message + "and " + buffets[-1].title() + "."  # fill in the last item with "and"
print(message)

buffets = ('fried chicken','cheese','mac n cheese','roast beef','dumpling')
message = "Today's buffet has "
for buffet in buffets[:-1]:
    message = message + buffet.title() +", "
message = message + "and " + buffets[-1].title() + "."
print(message)

