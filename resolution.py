import pandas as pd
from uncertainties.unumpy import uarray

resolution_file = './Images/20-01-25/resolution.csv'
df_ = pd.read_csv(resolution_file)
df_.index = [f'R{i}' for i in range(len(df_.index))]
particle_diameter = df_['particle_diameter(um)'].iloc[0] # um
df = df_.iloc[1:]

diameters_px = pd.concat([df['x_diameter'], df['y_diameter']])
particle_px = uarray(diameters_px, 20).mean()
resolution = particle_diameter / particle_px # um / px
print(particle_diameter)
df_.at['R0', 'particle_diameter(px)'] = str(particle_px)
df_.at['R0', 'resolution(um/px)'] = str(resolution)
print(df_)