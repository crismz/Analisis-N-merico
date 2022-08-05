import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#Ejercicio 1
def ilagrange(x,y,z):
    # la longitud de x,y deben tener la misma longitud
    assert(len(x) == len(y))
    m = []

    # Para calcular li
    for k in range(len(z)):
        sum = 0
        for i in range (0,len(x)):
            res = 1
            for j in range(0,len(x)):
                if i != j:
                    res = res * ((z[k]-x[j])/(x[i]-x[j]))
            sum = sum + res * y[i]
        m.append(sum)

    return m

#print(ilagrange([2,2.5,4],[1/2,2/5,1/4],[1,2,3]))


#Ejercicio 2

def horn_newton(zj,x,coefs):
	n = len(coefs)
	valor = coefs[n-1]
	for i in range(n-2,-1,-1):
		valor = coefs[i] + (zj - x[i])*valor
	return valor


def inewton(x,y,z):
    assert(len(x) == len(y))
    
    n = len(y)
    c = [[0.0]*m for m in range(n,0,-1)]

    for i in range(n):
        c[i][0] = y[i]

    for j in range (1,n):
        for i in range(0,n-j):
            c[i][j] = (c[i+1][j-1]-c[i][j-1]) / (x[i+j]-x[i]) 

    coefs = c[0]
    w = [horn_newton(zj,x,coefs) for zj in z]
    return w

#print(inewton([2,2.5,4],[1/2,2/5,1/4],[1,2,3]))


#Ejercicio 3
def fun(x): return 1/x 

z = [24/25 + j/25 for j in range (1,101)]
hx = inewton([1,2,3,4,5],[fun(1),fun(2),fun(3),fun(4),fun(5)],z)
y = [fun(i) for i in z]


#fig,ax = plt.subplots()

p = []
x = []
for i in range (1,6):
    p.append(1/i)
    x.append(i)
#ax.plot(x,p,'x')

#ax.plot(z,hx,label="Gráfico de p que interpola a f")
#ax.plot(z,y,label="Grafico de f = 1/x")
#ax.set_xlabel("Eje x")
#ax.set_ylabel("Eje y")
#ax.legend()
#ax.grid()
#plt.show()


#Ejercicio 4
def fun2(x): return 1/(1+25*x**2)

z = [i/100 for i in range (-100,101)]   #Espacios del eje x
f = [fun2(i) for i in z]    # función valuada en esos espacios

#xi del inciso a
def funxa(n):
    hxa = [(2*(i-1)/n)-1 for i in range (1,n+1)]
    return hxa

#xi del inciso b
def funxb(n):
    hxb = [math.cos(((2*i+1)/(2*n+2))*math.pi) for i in range (0,n+1)]
    return hxb

def lab3ej4():
    for i in range (1,16):
        # Crea una figura
        fig = 'fig' + str(i)
        ay = 'ay' + str(i)
        fig, ay = plt.subplots()
        ay.set_xlabel("Eje x")
        ay.set_ylabel("Eje y")
        
        # Grafica f
        ay.plot(z,f,label="f")
        

        # Grafica Pn
        xp = funxa(i)
        fp = [fun2(xp[j]) for j in range (0,i)]
        p = inewton(xp,fp,z)
        ay.plot(z,p,label="Pn")

        # Grafica Qn
        xp = funxb(i)
        fp = [fun2(xp[j]) for j in range (0,i+1)]
        q = inewton(xp,fp,z)
        ay.plot(z,q,label="Qn")

        ay.legend()
    plt.show()    

#lab3ej4()       # Ver github del año pasado, esta resuelto para que salgan los 15 graficos en una sola figura


#Ejercicio 5
def lab3ej5():
    datos = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB03/datos_aeroCBA.dat')
    years = datos[:,0]
    temps = datos[:,1]

    not_nan = ~np.isnan(temps)

    years_interp = years[not_nan]
    temps_interp = temps[not_nan]

    polinomio = interp1d(years_interp,temps_interp,kind = 'cubic',fill_value = 'extrapolate')

    years_plot = np.arange(1957,2016)
    temps_plot = polinomio(years_plot)
    plt.plot(years_plot, temps_plot)
    plt.plot(years_interp, temps_interp,'o')
    plt.grid()
    plt.show()

#lab3ej5()

#Ejercicio 6
def lab3ej6():
    x = [-3,-2,-1,0,1,2,3]
    y = [1,2,5,10,5,2,1]
    z = np.linspace(-3,3,100)
    plt.plot(x,y,'o')

    # Interpolación Lagrange
    il = ilagrange(x,y,z)
    plt.plot(z,il,label="Interpolación Lagrange")
    
    # Interpolación Newton
    ine = inewton(x,y,z)
    plt.plot(z,ine,label="Interpolación Newton")

    # Interpolación interp1d
    iinter = interp1d(x,y,kind='cubic')
    z_plot = iinter(z)
    plt.plot(z,z_plot,label="Interpolación interp1d")

    plt.legend()
    plt.grid()
    plt.show()
    
#lab3ej6()
#Es mas suave el polinomio de interp1d.


#Ejercicio 7            Revisar ejercicio, nose si esta bien
def dif_div_pol(x,y):
    assert(len(x) == len(y))
    
    n = len(y)
    c = [[0.0]*m for m in range(n,0,-1)]

    for i in range(n):
        c[i][0] = y[i]

    for j in range (1,n):
        for i in range(0,n-j):
            c[i][j] = (c[i+1][j-1]-c[i][j-1]) / (x[i+j]-x[i]) 

    coefs = c[0]
    x2 = coefs[2]
    x1 = coefs[2]* (-x[0]-x[1]) + coefs[1]
    x0 = coefs[2] * ((-x[0])*(-x[1])) + coefs[1] * (-x[0]) + coefs[0] 
    coefs2 = [x2, x1, x0] 
    return  coefs2


def rinterp(fun,x0,x1,x2,err,mit):
    k = 0
    while k < mit and abs(fun(x2)) > err:
        xs = [x0,x1,x2]
        fxs = [fun(x0),fun(x1),fun(x2)]
        q2 = dif_div_pol(xs,fxs)
        roots = np.roots(q2)
        a = abs(xs[2] - roots[1])
        b = abs(xs[2] - roots[0])
        if (a <= b):  
            x0,x1,x2 = x1,x2,roots[1]
        else:   
            x0,x1,x2 = x1,x2,roots[0]
        xs = [x0,x1,x2]
        k += 1
    return x2

# Ejemplo para comparar con rbisec
def fun_ej7_bisec(x): return x**2 - 3
#print(rinterp(fun_ej7_bisec,0,1,2,1e-5,100))