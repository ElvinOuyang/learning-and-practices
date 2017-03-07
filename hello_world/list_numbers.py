for value in range(1,5):  # "range()" starts from 1 and ends at 5, never containing 5 in return value
    print(value)

for value in range(1,6):
    print(value)

numbers = list(range(1,6))  # list() record each occurrence by range()
print(numbers)

print(list(range(1,20,3)))  # the third parameter in range() give the interval of the list