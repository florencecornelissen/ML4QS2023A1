from preprocessalldata import preprocessdatatotal
# names = ['Ivo', 'Joost', 'Flo']

# for name in names:
#     preprocessdatatotal(name)

indicesbeginFlo = [234, 193, 115, 186, 199, 360]
indicesendFlo = [2540, 3841, 3136, 2647, 3961, 1975]

indicesbeginIvo = [192, 183, 400, 346, 288, 337]
indicesendIvo = [4086, 4299, 3200, 4279, 2357, 2725]

indicesbeginJoost = [238, 258, 300, 300, 234, 342]
indicesendJoost = [1959, 1679, 1300, 2200, 1731, 1478]

participants = [{'name': 'Ivo', 'indicesbegin': indicesbeginIvo, 'indicesend': indicesendIvo},
               {'name': 'Flo', 'indicesbegin': indicesbeginFlo, 'indicesend': indicesendFlo},
               {'name': 'Joost', 'indicesbegin': indicesbeginJoost, 'indicesend': indicesendJoost}]

for participant in participants:
    preprocessdatatotal(participant['name'], participant['indicesbegin'], participant['indicesend'])

        