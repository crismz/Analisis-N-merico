import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt

x = [0,1.5,2,2.9,4,5.6,6,7.1,8.05,9.2,10,11.3,12]

fx = [0.1,0.2,1,0.56,1.5,2,2.3,1.3,0.8,0.6,0.4,0.3,0.2]

#inciso a)
plt.plot(x,fx,'o',label = "Datos")
plt.legend()
plt.show()



#inciso b)
def trapecio_adaptativo(x,y):
    if(len(x) != len(y)): 
        print("Error, los datos tienen que ser del mismo tamaño")
        return

    x_tam = len(x)
    S = 0                       # Va a guardar la aproximación de la integral
    for i in range (0,x_tam-1):
        aux = ((x[i+1] - x[i]) / 2) * (y[i] + y[i+1])         # Trapecio simple por cada subintervalo
        S = S + aux

    return S


#inciso c)
# Para nivelar a 0 metros, ahi que remover toda el área bajo la curva. 
# Entonces calculamos toda el área con trapecio_adaptativo y la multiplicamos por 10 que la profundidad y asi dan los metros cúbicos

m = trapecio_adaptativo(x,fx)
print(f"La cantidad aproximada de metros cúbicos de tierra que deben ser removidos para nivelar a 0 metros es {10*m}")