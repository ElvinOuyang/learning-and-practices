motorcycles = ['honda','yamaha','suzuki']
print(motorcycles)

"""
Replace
"""

motorcycles[0] = 'ducati'  # By redefining the value of the first element in the list, 'honda' is replaced by 'ducati'
print(motorcycles)

"""
Append
"""

motorcycles = ['honda','yamaha','suzuki']
print(motorcycles)
motorcycles.append('ducati')  # ".append()" adds extra items to the existing list at the end
print(motorcycles)

"""
Insert
"""

motorcycles = ['honda','yamaha','suzuki']
motorcycles.insert(0,'ducati')  # ".insert()" insert the element to the index position and move everything to the right
print(motorcycles)

"""
Delete
"""

motorcycles = ['honda','yamaha','suzuki']
print(motorcycles)
del motorcycles[0]  # note that the delete statement is a bit different than other methods
print(motorcycles)

"""
Pop
"""

motorcycles = ['honda','yamaha','suzuki']
print(motorcycles)
popped_motorcycle = motorcycles.pop()
print(motorcycles)
print(popped_motorcycle)

"""
The ".pop()" method pops the last element
 from the list and allows further actions with it
"""
motorcycles = ['honda','yamaha','suzuki']
first_owned = motorcycles.pop(0)
print('The fist motorcycle I owned was a ' + first_owned.title() + '.')

"""
Remove: removing an element by value (instead of index)
"""
motorcycles = ['honda','yamaha','ducati','suzuki']
print(motorcycles)
motorcycles.remove('ducati')
print(motorcycles)

motorcycles = ['honda','yamaha','ducati','suzuki']
too_expensive = 'ducati'
motorcycles.remove(too_expensive)  # the "remove()" method also works for variables
print(motorcycles)
print("\nA " + too_expensive.title() + " is too expensive for me.")

"""
remove() only removes the first occurrence of the value one specifies. A loop is needed if
one needs to remove all of occurrences.
"""

"""
Exercise
"""
  # Exercise 3.4
guests = ['rosalynn','steven','brian','brandon','matt','anthony','shane','dana','amine']
message = guests[0].title() + ",\nI hope you can join me this Friday for this event!\nBest,\nElvin"
print(message)
message = guests[-2].title() + ",\nI hope you can join me this Friday for this event!\nBest,\nElvin"
print(message)

  # Exercise 3.5
no_show = ['steven','matt','amine']
print(no_show)
guests[1] = "evan"
guests[4] = "leon"
guests[-1] = "john"
print(guests)

  # Exercise 3.6
print(guests)
additional_guests = ['jack','lily','chang','james']
print(additional_guests)
guests.insert(0,'jack')
guests.insert(2,'lily')
guests.append('chang')
print(guests)

