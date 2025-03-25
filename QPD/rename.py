import os
import pandas as pd

dirname = './data/11-03-25'
listdir = os.listdir(dirname)
skipnum = 17 # inclusive
for x in listdir:
	if 'labeling' in x: labeling = x
DS = [filename for filename in listdir if 'DS' in filename and not 'X_' in filename]
DS = [ds for ds in DS if int(ds.replace('.CSV', '').replace('DS', '')) >= skipnum]

labeling = pd.read_csv(os.path.join(dirname, labeling))

for ds, current, filename in zip(DS, labeling['current(mA)'], labeling['filename']):
	dsname = ds.split('.')[0]
	# 'x' marks measurement of unobstructed laser
	if 'x' in filename:
		dsname = f"{dsname}_{current}mA_reference.CSV"
	else:
		dsname = f"{dsname}_{current}mA.CSV"
	old_dsname = os.path.join(dirname, ds)
	new_dsname = os.path.join(dirname, dsname)
	os.rename(old_dsname, new_dsname)