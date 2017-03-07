players = ['charles','martina','michael','florence','eli']
print(players[0:3])  # indices 0:3 indicates 0,1,2 are selected
print(players[1:4])
print(players[:4])  # omitted first index means 0
print(players[2:])  # omitted last index means -1

print("Here are the first three players on my team:")
for player in players[:3]:
    print(player.title())



