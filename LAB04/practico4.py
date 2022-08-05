from cProfile import label
from calendar import c
from colorsys import ONE_SIXTH
import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt

def lab4ej1a():
    datos = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB04/datos1a.dat')
    #print(datos)
    x = datos[:,0]
    y = datos[:,1]
    n = len(x)

    sumx2 = np.sum(x**2)
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumxy = np.sum(x*y)

    b = (sumx2 * sumy - sumxy * sumx) / (n * sumx2 - sumx**2)
    a = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx**2)

    def fun(x):
        return a*x+b

    xs = np.linspace(0,5.5,2)

    plt.plot(x,y,'o')
    plt.plot(xs,fun(xs))
    plt.show()


def lab4ej1b():
    datos = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB04/datos1a.dat')
    #print(datos)
    x = datos[:,0]
    y = datos[:,1]
    n = len(x)

    ones = np.ones(n)

    sumx = np.dot(x,ones)
    sumy = np.dot(y,ones)
    sumxy = np.dot(x*y,ones)
    sumx2 = np.dot(x**2,ones)

    b = (sumx2 * sumy - sumxy * sumx) / (n * sumx2 - sumx**2)
    a = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx**2)

    def fun(x):
        return a*x+b

    xs = np.linspace(0,5.5,2)

    plt.plot(x,y,'o')
    plt.plot(xs,fun(xs))
    plt.show()


def lab4ej1c():
    x = np.linspace(0,10,20)

    def fun(x):
        return (3/4)* x  - (1/2)
    y = fun(x)

    v = np.random.randn(20)
    y_disp = y + v
    
    pol = np.polyfit(x,y_disp,1)
    
    print(f"Los coeficientes del ajuste son a = {round(pol[0],2)} y b = {round(pol[1],2)}")

    plt.plot(x,y,label="Recta original")
    plt.plot(x,np.polyval(pol,x),label="Aproximación lineal")
    plt.legend()
    plt.show()


def lab4ej2():
    cualF = int(input("Elija la función a utilizar, 0 si elije el arcoseno y 1 si elije el coseno.\n"))
    n = int(input("De el grado la aproximación lineal, que sea entre 0 y 5, incluyendolos.\n"))
    
    if (cualF == 0):
        x = np.linspace(0,1,50)
        def arcsen(x):
            return [math.asin(i) for i in x]
        y = arcsen(x)
    else: 
        x = np.linspace(0,4*math.pi,50)
        def cos(x):
            return [math.cos(i) for i in x]
        y = cos(x)


    def crearmat(n):
        if (n == 0): matrix = np.ones((50,1))
        else:   
            matrix = np.ones((50,n+1))
            for i in range (0,n+1): 
                matrix [:,i] = matrix[:,i] * x**i
        return matrix

    matrix = crearmat(n)                 #matriz con los x, x**2, etc.
    matrixT = matrix.transpose()         #matriz transpuesta

    Amatrix = np.dot(matrixT,matrix)     #matriz A trans multiplicada por A
    Bmatrix = np.dot(matrixT,y)          #matriz A trans multiplicada por B (donde estas los Yi)
    
    pol_sol = np.linalg.solve(Amatrix,Bmatrix)       #Conseguimos los coeficientes de la aproximación de grado n

    y_aprox = np.dot(matrix,pol_sol)      #Valores de y con polinomio de aproximación

     
    plt.plot(x,y,label="Recta original")
    plt.plot(x,y_aprox,label="Aproximación lineal")
    plt.legend()
    plt.show()


def lab4ej3a():
    datos = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB04/datos3a.dat')

    # y(x) = C*x**A    , aplico ln a ambos lados -->   ln y = ln(C) + A * ln(x)
    # y_p = ln(y)
    # x_p = ln(x) 
    # B = ln(C)
    # y_p = B + A * x_p

    x = datos[0]
    y = datos[1]
    
    x_p = np.log(x)
    y_p = np.log(y)

    coefs = np.polyfit(x_p, y_p, 1)
    
    A = coefs[0]
    C = np.exp(coefs[1])        # C = e**B

    print(f'Coef. A = {A}, Coef C = {C}.')

    plt.plot(x,y, 'o',label="F original")
    plt.plot(x, C * x ** A,label="Aprox. lineal")
    plt.legend()
    plt.show()


def lab4ej3b():
    datos = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB04/datos3b.dat')

    # y(x) = x/(A*x + B)    , aplico paso terminos multiplicando y dividiendo --> x/y =  A*x + B
    # y_p = x/y 
    # y_p = A*x + B 

    x = datos[0]
    y = datos[1]
    
    y_p = x/y

    coefs = np.polyfit(x, y_p, 1)
    
    A = coefs[0]
    B = coefs[1]        

    print(f'Coef. A = {A}, Coef B = {B}.')

    plt.plot(x,y, 'o',label="F original")
    plt.plot(x, x/(A*x+B),label="Aprox. lineal")
    plt.legend()
    plt.show()


def lab4ej4():

    datos = np.loadtxt("Escritorio/LaboratorioAnnumerico/LAB04/covid_italia.csv",delimiter=",", dtype=str)
    dates = datos[:, 0]
    cases = datos[:, 1].astype(int)

    x = np.array(range(1,len(dates)+1))
    y = cases

    # y = a* e ** (b*x)      --> aplico ln -->  ln(y) = ln(a) + b*x
    # y_p = ln(y), A = ln(a)  --> y_p = A + b * x
    
    y_p = np.log(y)         # y_p = ln(y)
    
    coefs = np.polyfit(x,y_p,1)

    a = np.exp(coefs[1])     # a = e**A

    def ajuste(n): return a * np.exp(coefs[0]*n)

    plt.plot(x,y, 'o',label="Casos Covid")
    plt.plot(x, ajuste(x),label="Aprox. lineal, Ajuste")
    plt.legend()
    plt.show()
