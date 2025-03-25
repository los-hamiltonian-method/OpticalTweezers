import os

dirpath = './data/11-03-25/renamed'
listdir = os.listdir(dirpath)

empty_reference = ''
for DS in listdir:
	if 'empty' in DS:
		empty_reference = DS
		print(empty_reference)
	current = DS.split('_')[1].replace('.CSV', '').replace('mA', '') # mA
	print(current)