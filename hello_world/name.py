name = "ada lovelace"
print(name.title())  # first letter of each word will be in upper case
print(name.upper())  # all letters will be in upper case
print(name.lower())  # all letters will be in lower case

first_name="ada"
last_name="lovelace"
full_name=first_name+" "+last_name  # use "+" for concatenation
message="Hello," + full_name.title() + "!"
print(message)

print("Python")
print("\tPython")
print("Languages:\nPython\nC\nJavaScript")  # \n is a new line and \t is a tab
print("Launguages:\n\tPython\n\tC\n\tJavaScript")

favorite_language= 'python '
print(favorite_language)
print(favorite_language.rstrip())  # rstrip() gets rid of whitespace on the right
favorite_language=favorite_language.rstrip()  # if need to update the string, redefine the variable
print(favorite_language)

favorite_language=' python'
print(favorite_language.lstrip())  # lstrip() gets rid of whitespace on the left
favorite_language=' python '
print(favorite_language.strip())  # strip() gets rid of whitespace on both sides

girl_friend_name="RoSalYnn yAng"
girl_friend_name=girl_friend_name.lower()
text="I love " + girl_friend_name + "!"
print(text.upper())


