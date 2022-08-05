import numpy as np
import math
import matplotlib.pyplot as plt


#Inciso a)

def regla_simpson(fun,a,b,N):
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
    return S

def fun(x): return 1/x

def integral_simpson(x):
    return regla_simpson(fun,1,x,100)

#Inciso b)
# integral_simpson(x) = 1
# integral_simpson(x) - 1 = 0
def fun2(x): return integral_simpson(x) - 1 

# Usamos el método de bisección por que sabemos que la raíz esta entre 2 y 3 ya queremos obtener una aproximación de e

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


hx,hy = rbisec(fun2,[2,3],1e-6,100)
aprox_e = hx[len(hx)-1]

print(f"La aproximación de e es {aprox_e}")

#Inciso c)

error_abs = abs(math.e - aprox_e)
print(f"El error absoluto de la aproximación con respecto a la constante e es {error_abs}")

x = np.linspace(1,8,100)

hyp = []
for i in range(0,len(hy)): hyp.append(hy[i]+1)

plt.plot(x,integral_simpson(x),label="Integral Simpson")
plt.plot(hx,hyp,'*',label="Puntos visitados con mét bisección")
plt.legend()
plt.grid()
plt.show()
