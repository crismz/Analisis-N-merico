import numpy as np
import sympy as sp
from scipy import integrate 
import math
import matplotlib.pyplot as plt
import time

#Ejercicio 1
def intenumcomp(fun,a,b,N,regla):
    if (regla == "simpson"):
        h = (b-a)/(2*N)
        sx0 = fun(a) + fun(b) # f(x0) + f(xn)
        sx1 = 0             #suma de f(x2j-1) de j=1 hasta n
        sx2 = 0             #suma de f(x2j) de j=1 hasta n
        x = a
        
        for j in range (1,2*N):
            x = x + h
            if (j % 2 == 0):
                sx2 = sx2 + fun(x)
            else:
                sx1 = sx1 + fun(x) 
        

        S = (h/3) * (sx0 + 2*sx2 + 4*sx1)
        #print(S)
        return S

    elif (regla == "trapecio"):
        h = (b-a) / N
        sx0 = fun(a) + fun(b)  # f(x0) + f(xn)
        sx1 = 0             #suma de f(xj) de j=1 hasta n-1
        x = a

        for j in range (1,N):
            x = x + h
            sx1 = sx1 + fun(x)
        
        S = (h/2) * (sx0 + 2*sx1)
        #print(S)
        return S

    elif (regla == "pm"):
        h = (b-a) / (N+2)
        sx0 = 0           # suma de f(x2j) de j=0 hasta n/2
        for j in range (0,int(N/2)):
            x = a
            x = x + 2*j * h
            sx0 = sx0 + fun(x)
        
        S = (2*h) * (sx0)
        #print(S)
        return S

    else:
        print("La regla dada no es ninguna de las tres para selecionar")
        return

    
def funejemplo(x):
    return math.exp(-x**2)


def funej2(x):
    return math.exp(-x)

def lab5ej2():
    realvalue = integrate.quad(funej2,0,1)
    
    intervalos = [4,10,20]

    print(f"El valor real de la integral es {realvalue[0]}")
    
    for N in intervalos:
        simp = intenumcomp(funej2,0,1,N,"simpson")
        print(f"El valor de simpson con {N} subintervalos es {simp} y su error absoluto es {realvalue[0]-simp}")
        trape = intenumcomp(funej2,0,1,N,"trapecio")
        print(f"El valor de trapecio con {N} subintervalos es {trape} y su error absoluto es {realvalue[0]-trape}")
        pm = intenumcomp(funej2,0,1,N,"pm")
        print(f"El valor de punto medio con {N} subintervalos es {pm} y su error absoluto es {realvalue[0]-pm}")
   

#Ejercicio 3
def senint(x):
    def fun(w):
        return math.cos(w)

    y = []
    for i in x:
        N = i * 10
        N = math.ceil(N)
        y.append(intenumcomp(fun,0,i,N,"trapecio"))
    return y


def lab5ej3():
    x = np.linspace(0,2*math.pi,13)            
    y_sin = np.sin(x)

    plt.plot(x,y_sin,label = "Función seno")
    plt.plot(x,senint(x),label = "Función senint")
    plt.legend()
    plt.show()



#Ejercicio 4

def lab5ej4():

    # inciso a)
    def funa(x):
        return x * math.exp(-x)
        
    #trapecio
    trapea = intenumcomp(funa,0,1,130,"trapecio")    
    err = (trapea - (1-2/math.exp(1)))
    print(f"Para el inciso a, para que la aproximación de la integral usando trapecio tenga un error menor a 10**-5 necesitamos 130 subintervalos y el resultado es {trapea} con error {err}")

    #simpson
    simpson = intenumcomp(funa,0,1,7,"simpson")    
    err = (simpson - (1-2/math.exp(1)))
    print(f"Para el inciso a, para que la aproximación de la integral usando simpson tenga un error menor a 10**-5 necesitamos 7 subintervalos y el resultado es {simpson} con error {err}")

    
    # inciso b)
    def funb(x):   
        return x * math.sin(x)

    #trapecio
    trapea = intenumcomp(funb,0,1,130,"trapecio")    
    err = (trapea - (-math.cos(1) + math.sin(1)))
    print(f"Para el inciso b, para que la aproximación de la integral usando trapecio tenga un error menor a 10**-5 necesitamos 130 subintervalos y el resultado es {trapea} con error {err}")

    #simpson
    simpson = intenumcomp(funb,0,1,7,"simpson")    
    err = (simpson - (-math.cos(1) + math.sin(1)))
    print(f"Para el inciso b, para que la aproximación de la integral usando simpson tenga un error menor a 10**-5 necesitamos 7 subintervalos y el resultado es {simpson} con error {err}")

    # inciso c)
    def func(x):   
        return (1 + x**2)**(3/2)

    #trapecio
    trapea = intenumcomp(func,0,1,231,"trapecio")    
    err = (trapea - 1.56795196220879)
    print(f"Para el inciso c, para que la aproximación de la integral usando trapecio tenga un error menor a 10**-5 necesitamos 231 subintervalos y el resultado es {trapea} con error {err}")

    #simpson
    simpson = intenumcomp(func,0,1,9,"simpson")    
    err = (simpson - 1.56795196220879)
    print(f"Para el inciso c, para que la aproximación de la integral usando simpson tenga un error menor a 10**-5 necesitamos 9 subintervalos y el resultado es {simpson} con error {err}")



#Ejercicio 5

def lab5ej5():
    def funa(x):
        return math.exp(-x**2)
    
    realvaluea = integrate.quad(funa,-math.inf,math.inf)
    print(f"El valor de la integral del inciso a es {realvaluea[0]}")
    
    def funb(x):
        return x**2 * math.log(x+(x**2+1)**(1/2))
    
    realvalueb = integrate.quad(funb,0,2)
    print(f"El valor de la integral del inciso b es {realvalueb[0]}")

#Ejecicio 6

def pendulo(l,alpha):
    alpha_rad = alpha * math.pi / 180

    fun = lambda theta: 1 / (np.sqrt(1 - np.sin(alpha_rad/2)**2 * np.sin(theta)**2))

    valor_integral = intenumcomp(fun,0,math.pi/2,2**10,"simpson")

    periodo = 4 * np.sqrt(l/9.8) * valor_integral

    print(f"Tenemos un periodo de {periodo}")

    return periodo



 #Ejercicio 7

def cuad_adap_simpson(fun,a,b,err):
    q = intenumcomp(fun,a,b,2,'simpson')    # S(a,b)

    c = (a + b)/2
    q1 = intenumcomp(fun,a,c,2,'simpson')   # S(a,c)
    q2 = intenumcomp(fun,c,b,2,'simpson')   # S(c,b)

    if (abs(q - q1 - q2) < 15 * err):       
        I = q1 + q2
    else: 
        q1 = cuad_adap_simpson(fun,a,c,err/2)
        q2 = cuad_adap_simpson(fun,c,b,err/2)
        I = q1 + q2

    return I

def lab5ej7():
    fun = lambda x : x*np.exp(-x**2)   # Funcion a integrar

    err = 1e-15                         # Error elegido

    I_exacta = 0.5*(1-np.exp(-1))       # Valor exacto de la integral de la función

    start = time.time()                 # Start timer

    I_recursiva = cuad_adap_simpson(fun,0,1,err)       # Resultado de la integración de simp_cuad_adaptiva

    print(f"Simpson adaptativo recursivo demoró {time.time()-start} y calculó {I_recursiva}") 

    cota_d4f = 156                      

    N_simpson = int(np.ceil((cota_d4f/(err*180))**(1/4)))              

    start = time.time()

    I_compuesta = intenumcomp(fun,0,1,N_simpson,"simpson")

    print(f"Simpson compuesta demoró {time.time()-start} y calculó {I_compuesta}")
