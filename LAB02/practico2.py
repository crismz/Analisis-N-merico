#Libraries
from re import X
import re
import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
# Ejercicio 1 Metodo de biseccion

def rbisec(fun,I,err,mit):
    u = fun(I[0])     #f(a)
    v = fun(I[1])     #f(b)
    e = I[1] - I[0]
    hx = []
    hf = []

    #print(f"f(a) = {u}, f(b) = {v}")

    if np.sign(u) == np.sign(v):
        return hx, hf

    for k in range(mit):
        e = e/2
        c = I[0] + e
        w = fun(c)   #f(c)
        if abs(e) < err:
            #print(hx)
            #print(hf)
            #print(k)
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

# Ejemplo: 
def fun(x):
    x = x**2 - 3 
    return x

#print(rbisec(fun,[-2,0],1e-5,100))
#Ejercicio 2 Usos de rbisec
fig,ax = plt.subplots()

# a)   Son necesarias 15 iteraciones
def fun_lab2ej2a(x):
    return math.tan(x) - 2*x

hx,hy = rbisec(fun_lab2ej2a,[0.8,1.4],1e-5,100)

"""
x = np.linspace(0,2,200)
ax.plot(hx,hy,'*',label="Estimaciones ej2a")
ax.plot(x,np.tan(x)-2*x,label="Fun ej2a")
"""

# b)
def fun_lab2ej2b(x):
    x = x**2-3
    return x

hx,hy = rbisec(fun_lab2ej2b,[1,2],1e-5,100)

"""
ax.plot(hx,hy,'*',label="Estimaciones ej2b")
ax.plot(x,x**2 - 3,label="Fun ej2b")
ax.set_xlabel("Eje x")
ax.set_ylabel("Eje y")
ax.legend()
#plt.show()
"""

#Ejercicio 3 Metodo de Newton
def rnewton(fun,x0,err,mit):
    v,dv = fun(x0)
    hx = []
    hf = []
    for k in range(0,mit):
        x1 = x0-v/dv
        v,dv = fun(x1)
        #print(k,x1,v,dv)      Imprime los valores de la iteracion, el de la proxima aproximacion (x1), el f(x1) y f'(x1)
        if abs(x1-x0)/abs(x1) < err or abs(v) < err:
            #print(hx)
            #print(hf)
            return hx,hf
        hx.append(x0)
        hf.append(v)
        x0 = x1
    return hx,hf

# Funcion prototipo para valuar en f y su derivada
def funYDiff_Prototipo(v):
    x = sp.Symbol('x')
    fun = x**2
    diff = sp.diff(fun,x)
    a = (fun.subs(x,v))
    b = (diff.subs(x,v))
    return a,b 
    
# Ejemplo
def funYDiff(v):
    x = sp.Symbol('x')
    fun = x**2 - 17
    diff = sp.diff(fun,x)
    a = (fun.subs(x,v))
    b = (diff.subs(x,v))
    return a,b 


#print(rnewton(funYDiff,4,1e-15,10))

#Ejercicio 4
def funYDiff_Ej4(v,a):
    x = sp.Symbol('x')
    fun = x**3 - a
    diff = sp.diff(fun,x)
    a = (fun.subs(x,v))
    b = (diff.subs(x,v))
    return a,b 


def aprox_raiz(a):
    assert(a > 0)
    def f(x): 
        return (x**3 - a) 
    def df(x): 
        x = 3*x**2-a
        assert(x != 0)
        return x
    def fun(x): 
        return (f(x),df(x))
    x0 = a/3    
    hx, _ = rnewton(fun,x0,10**(-6), 1000) 
    
    return hx

#print(aprox_raiz(3/2))

#Ejercicio 5
def ripf(fun,x0,err,mit):
    hx = []    
    i = 1
    while (i <= mit):
        x1 = fun(x0)
        if abs(x1 - x0) < err:
            hx.append(x1)
            return hx     
        x0 = x1
        hx.append(x1)
        #print(i)
        i += 1
    return hx       

#Ejercicio 6
def fun_lab2ej6(x0):
    return 2**(x0-1)

hx = ripf(fun_lab2ej6,1.5,1e-10,100)
#print(hx)   

# La derivada nos da 2**(x-1) * ln(2)
# Para cualquier x < 1, esta derivada es < 1
# Ademas, la funcion es exponencial, por lo que tiende a 0(+) cuando x tiende a -Inf.
# Concluyo entonces que la funcion converge con cualquier x0 en (-Inf,1)


#Ejercicio 7
def lab2ej7bisec(x):
    def ec(y): return y - math.exp(-(1-x*y)**2)
    hy, _ = rbisec(ec,[0,1.5],1e-5,200)
    return hy[-1]

#print(f"u(0.7)={lab2ej7bisec(0.7)}, segun Biseccion")


def lab2ej7newton(x):
    def ec(y): return y - math.exp(-(1-x*y)**2)
    def ecdiff(y): return 1 + 2*math.exp(-(1-x*y)**2) * (x-(x**2)*y)
    def f(y): return (ec(y),ecdiff(y))

    # Al despejar la ecuacion y = e**(-(1-xy)**2),
    # Obtenemos ln y = -((1-xy)**2), osea que ln y<0
    # Por lo tanto y esta entre 0 y 1.
    # Uso entonces y0 = 0.5

    y0 = 0.5
    hy,_ = rnewton(f,y0,1e-5,100)
    return hy[-1]

#print(f"u(0.7)={lab2ej7newton(0.7)}, segun Newton")


def lab2ej7ipf(x):
    def ec(y): return math.exp(-(1-x*y)**2)

    # Al despejar la ecuacion y = e**(-(1-xy)**2),
    # Obtenemos ln y = -((1-xy)**2), osea que ln y<0
    # Por lo tanto y esta entre 0 y 1.
    # Uso entonces y0 = 0.5

    y0 = 0.5
    hy = ripf(ec, y0, 1e-5, 200)
    return hy[-1]

#print(f"u(0.7)={lab2ej7ipf(0.7)}, segun Iteracion de punto fijo")
    

#Ejercicio 8
def derivadafea(x): return (x*(1/math.cos(x))**2 - 2 * math.tan(x)) / x**3

def derivadafea2(x): return 2*(x**2*(1/math.cos(x))**2 *
                               math.tan(x)-2*x*(1/math.cos(x))**2+3*math.tan(x))/x**4


def ffea(x): return (derivadafea(x), derivadafea2(x))


hx, hf = rnewton(ffea, 1.5, 1e-10, 200)
#print(hf)

#print(f"El minimo de la funcion entre 0 y medio pi es {hx[-1]}, segun el metodo de Newton")


#Ejercicio 9

# Funcion que da 0 cuando el diametro del las aspas es el necesario para
# tener un output de 500W
def fun_lab2ej9(d): return (d**2)*(296.29629629629636) - 500
def fun_lab2ej9prim(d): return 2*d*(296.29629629629636)
def f(x): return (fun_lab2ej9(x), fun_lab2ej9prim(x))


hx, _ = rnewton(f, 1.3, 1e-5, 200)

#print(f"Es necesario un diametro de {hx[-1]} para tener 500W de output segun el metodo de Newton")