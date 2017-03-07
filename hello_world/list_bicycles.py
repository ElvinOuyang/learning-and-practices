bicycles = ['trek','cannondale','redline','specialized']
print(bicycles)  # printing the list altogether will include all the operators
print(bicycles[0])  # use [number] to indicate the index of the selected element
print(bicycles[0].upper())  # the first element in the list takes position [0]
print(bicycles[-1])  # [-1] is the index of the LAST element in a list
print(bicycles[-3])  # [-3] is the third last element in a list

message = "My first bicycle was a " + bicycles[0].title() + "."
print(message)

