#Ejercicio 3 (overflow y underflow)
from cmath import sqrt
import math
import random
from re import S
import re
from tkinter import Y

def overflow(x):
    while not math.isinf(x):
        x = x**2
        print(x)

def underflow(x):
    while x != 0:
        y = x
        x = x/2
    print(y)

#Ejercicio 4
def errorRedon(x):
    x = 0
    while x < 10:
        x = x + 0.1
        print(x)
     # al ir sumando 0.1 se va perdiendo el valor y por eso no llega a 10 exacto por como guarda la informarción la compu. (errores de redondeo). No pasa con 0.5

def  sinError(x):
    x = 0
    while x != 10:
        x = x + 0.5
        print(x)

#Ejercicio 5
def fact(x):
    if x==0:
        x = 1
    else:
        x = x * fact(x-1) 
    return(x)

def fact2(x):
    x = math.factorial(x)
    print(x)

#Ejercicio 6
def mayor(x,y):
    if x > y:
        print(f"{x} es mayor")
    elif x < y:
        print(f"{y} es mayor")
    elif x == y:
        print(f"{x} y {y} son iguales")

#Ejercicio 7
def potencia(x,n):
    base = x
    print(f"1° Potencia: {x}")
    if n == 0:
            x = 1
    else:
        for i in range (1,n):  
            x = x*base
            if i < 6:
                print(f"{i}° Potencia: {x}")
    print(x)


#Ejercicio 8
def mala(a,b,c):
    disc = b**2 - 4 * a * c
    if disc < 0:
        print("No existen raíces Reales")
    elif disc == 0:
        n = (-b + (disc)**(1/2))/(2*a)
        print(f"{n} Es la única solución real")
        return n
    elif disc > 0:
        n = (-b + (disc)**(1/2))/(2*a)
        m = (-b - (disc)**(1/2))/(2*a)
        print(f"{n} y {m} son las soluciónes reales")
        return n,m

def buena(a,b,c):
    disc = b**2 - 4 * a * c
    if disc < 0:
        print("No existen raíces Reales")
    elif disc == 0:
        n = (-b + (disc)**(1/2))/(2*a)
        print(f"{n} Es la única solución real")
        return n
    elif disc > 0:
        if b < 0:
            n = (-b + (disc)**(1/2))/(2*a)
        else:
            n = (-b - (disc)**(1/2))/(2*a)
        m = c/(a*n)

        print(f"{m} y {n} son las soluciónes reales")
        return n,m

def llamadaSolucionCuadratica ():
    print("De el coeficiente de mayor grado")
    a = float(input())
    print("De el coeficiente de un grado")
    b = float(input())
    print("De el término independiente")
    c = float(input())
    mala(a,b,c)
    buena(a,b,c)

#Ejercicio 9
def horn(coefs,x):
    b = coefs[0]
    for i in range (1,len(coefs)):
        b = coefs[i] + x * b
    return b

#Ejercicio 10
def sonReciprocos(x,y):
    if x*y == 1:
        res = True
    else:
        res = False
    return res

"""
for i in range(100):
    x = 1 + random.random()
    y = 1/x
    if not sonReciprocos(x,y):
        print(x)
"""

#Ejercicio 11

def f(x):
    n = sqrt(x**2+1)-1
    return n

def g(x):
    n = (x**2) / (sqrt(x**2+1)+1)
    return n

# Resulta mas confiable  la funcion g(x), por no haber resta da menos errores
"""
for i in range(20):
    x = 8 **-i
    print(f"f(x)={f(x)}, g(x)={g(x)}")
"""

#Ejercicio 12
def sonOrtogonales(x,y):
    if (x[0]*y[0] + x[1]*y[1] == 0):
        return True
    else:
        return False


"""
x = [1,1.1024074512658109]
y = [-1,1/x[1]]
if not sonOrtogonales(x,y):
    print("Algo salio mal....")
"""