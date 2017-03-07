"""
Sort Permanently
"""
cars = ['bmw','audi','toyota','subaru']
cars.sort()  # permanently sort the order to alphabetic order
print(cars)
cars.sort(reverse=True)  # reverse alphabetic order
print(cars)

"""
sort just for display
"""

cars = ['bmw','audi','toyota','subaru']
print("\nHere is the original list:")
print(cars)
print("\nHere is the sorted list:")
print(sorted(cars))  # the "sorted()" function temporarily sort the list for the current action
print("\nHere is the original list again:")
print(cars)

"""
reverse order of the list
"""
cars = ['bmw','audi','toyota','subaru']
print(cars)
cars.reverse()  # reverse() sort the list backward by simply reverse the sequence
print(cars)
print("This list has " + str(len(cars)) + " brands.")  # Note to use "str()" for numbers to transfer them into strings




