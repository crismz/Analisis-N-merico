#En cada ejercicio dice que hacer para poder realizar la ejecución. En algunos ahi 
# hacer el print y la llamada de la función, en otros solo descomentar la llamada.


import math
import matplotlib.pyplot as plt
import numpy as np

# Ejercicio 1: Hacer la llamda de la función y un print
def serie_seno(x):
    sum = 0
    hx = []
    for i in range (0,5):
        term = (((-1)**i) / math.factorial(2*i+1)) * x**(2*i+1)
        sum = sum + term
        hx.append(term)
    
    return hx,sum

# La función da hx que es la lista con las primeras 5 iteraciones sin sumar. 
#   Y sum que es la suma total de las 5 iteraciones
# Ejemplo de ejecución:
#   hx,sum = serie_seno(5)
#   print(f"Las primeras 5 iteraciones  son {hx}")
#   print(f"La suma de las primeras 5 iteraciones es {sum}")


# Ejercicio 2: Descomentar el plt.show y plt.grid

listofx = []
listofy = []
for i in range(0,641):
    x = 0.01 * i
    listofx.append(x)
    _,y = serie_seno(0.01*i)   
    listofy.append(y)
    

fig,ax = plt.subplots()
x = listofx
y = listofy
ax.plot(x,y,label="Gráfico de f")
ax.set_xlabel("Eje x")
ax.set_ylabel("Eje y")
ax.legend()
#plt.grid()
#plt.show()

#Del gráfico vemos que en el intervalo [0,6.4] tiene dos raices positivas, 
#   una alrededor de 3.20 y otra alrededor de 4.95.

# Ejercicio 3: Descomentar los print

# Met. de bisección
def rbisec(fun,I,err,mit):
    u = fun(I[0])     #f(a)
    v = fun(I[1])     #f(b)
    e = I[1] - I[0]
    hx = []
    hf = []

    if np.sign(u) == np.sign(v):
        return hx, hf

    for k in range(mit):
        e = e/2
        c = I[0] + e
        w = fun(c)   #f(c)
        if abs(e) < err:
            return hx,hf
        elif np.sign(w) != np.sign(u):     #f(a)*f(c) < 0
            I[1] = c
            v = w
        else:                       #f(a)*f(c) > 0 
            I[0] = c
            u = w
        hx.append(c)
        hf.append(w)
    return hx,hf

def fun(x):
    _,sum = serie_seno(x)
    return sum

#Del gráfico del ej 2 podemos estimar que una de las raices esta entre [3,3.5] y la otra entre [4.5,5.5]

# Primera Raíz

hx,hf = rbisec(fun,[2.5,3.5],1e-5,100)
#print(hx,hf)
#print(f"La primera raíz positiva que esta enter 3 y 3.5 es {hx[-1]}") 

# Segunda Raíz

hx,hf = rbisec(fun,[4.5,5.5],1e-5,100)
#print(hx,hf)
#print(f"La segunda raíz positiva que esta enter 4.5 y 5.5 es {hx[-1]}") 


# Ejercicio 4

def rsteffensen(fun,x0,err,mit):
    v = fun(x0)
    hx = []
    hf = []
    for k in range(0,mit):
        x1 = x0 - (v**2)/(fun(x0+fun(x0))-v)
        v = fun(x1)
        hx.append(x1)
        hf.append(v)
        if abs(v) < err:
            return hx,hf
        x0 = x1
    return hx,hf


# Ejercicio 5: Descomentar los print

# Primera raiz

hx1,hf1 = rsteffensen(fun,3,1e-5,100)
#print(hx1,hf1)
#print(f"La primera raiz positiva es {hx1[-1]}")
#print(f"Realiza {len(hx1)} iteraciones")


# Segunda raiz

hx2,hf2 = rsteffensen(fun,6,1e-5,100)
#print(hx2,hf2)
#print(f"La segunda raiz positiva es {hx2[-1]}")
#print(f"Realiza {len(hx2)} iteraciones")


# Busqueda con 4.5

hx3,hf3 = rsteffensen(fun,4.5,1e-5,100)
#print(hx3,hf3)
#print(f"Realiza {len(hx3)} iteraciones")

#  Viendo el arreglo vemos que a partir de la segunda iteración todas las iteraciones siguiente se mantienen
#   alrededor del mismo valor (converge a ese valor) que no es una raíz. Viendo el gráfico en el punto que 
#   oscila (-8.67) vemos que la función NO está definida para ese valor de x. Por lo tanto, con el valor 4.5, el método
#   falla. 

