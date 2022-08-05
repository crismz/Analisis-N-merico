import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,lagrange

datos = np.genfromtxt('Escritorio/LaboratorioAnnumerico/Parcial2/irma.csv', delimiter=',')
horas = datos[:,0]
longitud = datos[:,1]
latitud = datos[:,2]

# inciso a)
plt.plot(longitud,latitud,'o',label="Puntos de los datos")



#inciso b)
long_lagrange = lagrange(horas,longitud)                
long_splinecubic = interp1d(horas,longitud,'cubic')

lat_lagrange = lagrange(horas,latitud)
lat_splinecubic = interp1d(horas,latitud,'cubic')

horas_24 = np.arange(0,25,1)

print("Estimaciones de longitud cada una hora según spline cúbico:\n")
print(long_splinecubic(horas_24))
print("\nEstimaciones de latitud cada una hora según spline cúbico\n")
print(lat_splinecubic(horas_24))




fig,ax = plt.subplots()
ax.plot(long_lagrange(horas_24),lat_lagrange(horas_24),'o')
ax.plot(long_lagrange(horas_24),lat_lagrange(horas_24),label = "Lagrange")

ax.plot(long_splinecubic(horas_24),lat_splinecubic(horas_24),'o')
ax.plot(long_splinecubic(horas_24),lat_splinecubic(horas_24),label = "Spline")

ax.set_xlabel("Eje x")
ax.set_ylabel("Eje y")
ax.grid()
plt.legend()
plt.show()