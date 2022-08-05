import numpy as np
import sympy as sp
from scipy import integrate 
import math
import matplotlib.pyplot as plt
import time

#Ejercicio 1

def soltrinf(A,b):
    if (np.linalg.det(A) == 0):
        print("La matriz ingresada debe ser NO singular")
        return 

    tam_mat = np.shape(A)
    n = tam_mat[0]

    x = np.arange(n,dtype=float)
    for i in range (0,n):
        sum = 0 
        if (i != 0):
            for j in range (0,i):
                sum = A[i][j] * x[j] + sum

        x[i] = (b[i] - sum) / A[i][i]

    return x


def soltrsup(A,b):
    if (np.linalg.det(A) == 0):
        print("La matriz ingresada debe ser NO singular")
        return 

    tam_mat = np.shape(A)
    n = tam_mat[0]

    x = np.arange(n,dtype=float)
    for i in range (n-1,-1,-1):
        sum = 0 
        for j in range (i+1,n):
            sum = A[i][j] * x[j] + sum

        x[i] = (b[i] - sum) / A[i][i]

    return x


# Ejercicio 2
# a)
def egauss(A,b):
    if (np.linalg.det(A) == 0):
        print("La matriz ingresada debe ser NO singular")
        return 

    tam_mat = np.shape(A)
    n = tam_mat[0]

    for k in range(0,n-1):
        for i in range(k+1,n):
            if (A[k][k] == 0): return

            m = A[i][k] / A[k][k]

            for j in range(k+1,n+1):
                A[i][j-1] = A[i][j-1] - m*A[k][j-1]
            
            b[i] = b[i] - m*b[k]

    return A,b


# b)
def soleg(A,b):
    U,y = egauss(A,b)
    x = soltrsup(U,y)
    return x

A = [[5,4,3],[2,7,4],[8,13,5]]
b = [53,46,99]
print(soleg(A,b))


# Ejercicio 3
