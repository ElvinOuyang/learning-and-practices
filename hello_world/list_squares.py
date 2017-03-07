"""
for loop
"""
squares = []  # create an empty list
for value in range(1,11):  # start a loop from 1 to 10
    square = value**2  # calculate each element
    squares.append(square)  # add the new element to the list, repeat

print(squares)

squares = []
for value in range(1,11):
    squares.append(value**2)
print(squares)

"""
Simple List Calculation
"""

print(min(squares))  # simple statistical descriptive analysis
print(max(squares))
print(sum(squares))

"""
List Comprehension
"""

squares = [value**2 for value in range(1,11)]
  # list_name = [expression for loop]
print(squares)

"""
Exercise
"""

# 4.3
numbers_1 = [value for value in range(1,21)]
print(numbers_1)

# 4.4
numbers_2 = []
for value in range(1,10001):
    number = value
    numbers_2.append(number)
print(numbers_2)

# 4.5
numbers_3 = [value for value in range(1,1000001)]
print(min(numbers_3))
print(max(numbers_3))
print(sum(numbers_3))

# 4.6
numbers_4 = []
for value in range(1,20,2):
    numbers_4.append(value)
for x in numbers_4:
    print(x)

# 4.7
threes = [value for value in range(3,31,3)]
for x in threes:
    print(x)

# 4.8
cubes = [y**3 for y in range(1,11)]
for x in cubes:
    print(x)

# 4.9
cubes=[]
for y in range(1,11):
    cubes.append(y**3)
print(cubes)
