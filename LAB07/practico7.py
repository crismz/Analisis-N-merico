import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#Ejercicio 1
### PLANTEO DEL PROBLEMA
### x = cuanto tengo que comprar de T1 y T2,
### (x1 := cantidad de T1, x2 := cantidad de T2)
### c = valores de cada kilo de fertilizante (10, 8)
### 3 restricciones, minimo de P, N y K
### x1 * 3 + x2 * 2 >= 3 (P)
### x1 * 1 + x2 * 3 >= 1.5 (N)
### x1 * 8 + x2 * 2 >= 4 (K)

### Se transforma (para entrar en scipy)
### x1 * -3 + x2 * -2 <= -3 (P)
### x1 * -1 + x2 * -3 <= -1.5 (N)
### x1 * -8 + x2 * -2 <= -4 (K)
###         [-3, -2]
### A_ub =  [-1, -3]
###         [-8, -2]
### b_ub = [-3, -1.5, -4]
### No tenemos restricciones de igualdad
### Tenemos cotas (0, inf) para nuestras variables, esto es la cota por defecto en Scipy

def ej1():
    costo = np.array([10, 8], dtype="float")
    mat_des =  np.array([
        [-3,-2],
        [-1,-3],
        [-8,-2],
    ], dtype="float")
    vec_des = np.array([-3,-1.5,-4], dtype="float")

    res = optimize.linprog(c=costo, A_ub=mat_des, b_ub=vec_des)
    solucion = res.x

    print(f"Encontro solución {res.success}")
    print(f"Para minimizar el costo, necesitamos comprar {solucion[0]} de T1 y {solucion[1]}  de T2")
    print(f"Lo que se gasta en total por cada 10 metros cuadrados es {res.fun}")

    # para graficar
    # y = 1.5 - 1.5 x
    # y = 0.5 - 1/3 x
    # y = 2 - 4 x

    x = np.linspace(0, 3, 100)
    y1 = 1.5 - 1.5 * x
    y2 = 0.5 - 1/3 * x
    y3 = 2 - 4 * x
    curva_region = np.maximum(np.maximum(y1, y2), y3)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.fill_between(x, curva_region, 3, alpha=0.2)
    plt.plot(solucion[0], solucion[1], '*')
    plt.xlim([0, 3])
    plt.ylim([0, 3])
    plt.show()


#Ejercicio 2

# restricciones 
# 50*x + 24*y <= 2400 (1)
# 30*x + 33*y <= 2100 (2)

def ej2():
    x = np.linspace(0,50,100)

    y1 = (2400 - 50*x)/24
    y2 = (2100 - 30*x)/33

    restricciones = np.minimum(y1,y2)

    restricciones = np.maximum(restricciones,0)

    # plt.plot(x,y1)
    # plt.plot(x,y2)

    # plt.fill_between(x,y1,alpha=0.5)
    # plt.fill_between(x,y2,alpha=0.5)

    print("Aproximadamente la curva de nivel que da con la solución es la de 66.66, en el punto (30.7,35.6) (aproximadamente)")

    ## CURVAS DE NIVEL ##

    levels = np.linspace(40,80,10)

    for level in levels:
        # x+y = level
        y = level - x
        plt.plot(x,y,label=f"nivel={level}")

    plt.fill_between(x,restricciones,alpha=0.3)
    plt.plot(30.7, 35.6, '*')

    plt.grid()
    plt.legend()

    plt.show()

#Ejercicio 3
# a) La función objetivo es maximizar z = 25 * x1 + 20 * x2
# donde x1 y x2 son las medicinas 1 y 2 respectivamente.
# Las restricciones son:
#   3*x1 + 4*x2 <= 25
#   2*x1 + x2 <= 10
#   x1,x2 >= 0
    
def ej3():
# b)
#   Para gráficar:
#       x2 = (25 - 3*x1) / 4 
#       x2 = 10 - 2*x1

    x = np.linspace(0, 8, 100)
    y1 = (25 - 3 * x) /4
    y2 = 10 - 2 * x
    curva_region = np.minimum(y1, y2)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.fill_between(x, curva_region, 0, alpha=0.2)
    plt.xlim([0, 8])
    plt.ylim([0, 10])
    plt.show()

# c)
#   Transformar la función objetivo en minimizar
#   minimizar -z = -25 * x1 - 20 * x2
#
    c = np.array([-25, -20], dtype="float")
    A_ub =  np.array([
        [3,4],
        [2,1],
    ], dtype="float")
    b_ub = np.array([25,10], dtype="float")

    res = optimize.linprog(c, A_ub, b_ub)
    solucion = res.x

    print(f"Encontro solución {res.success}")
    print(f"Para maximizar la curación de salud, necesitamos crear {round(solucion[0])} de medicamento 1 y {round(solucion[1])} de medicamento 2")
    print(f"La curación total es {round(-res.fun)}")


#Ejercicio 4
#   Funcion objetivo: Maximizar z = 7*x1 + 4*x2 + 3 *x3
#  donde x1,x2,x3 son las cervezas Rubia, Negra y Baja respectivamente.
#
#  Restricciones:
#   x1 + 2*x2 + 2*x3 <= 30
#   2*x1 + x2 + 2*x3 <= 45
#   x1,x2,x3 >= 0

def ej4():
    # Se multiplica por -1 para que sea minimizar
    # -z = -7*x1 - 4*x2 - 3*x3
    c = np.array([-7, -4, -3], dtype="float")
    A_ub =  np.array([
        [1,2,2],
        [2,1,2]
    ], dtype="float")
    b_ub = np.array([30,45], dtype="float")

#   Si quisieramos poner los bounds:
#   x_bound = (0, None)
#   y_bound = (0, None)
#   z_bound = (0, None)
#   res = optimize.linprog(c, A_ub, b_ub, bounds = [x_bound, y_bound, z_bound])

    res = optimize.linprog(c, A_ub, b_ub)
    solucion = res.x

    print(f"Encontro solución {res.success}")
    print(f"Para maximizar la ganancia, necesitamos fabricar {round(solucion[0])} de cerveza Rubia, {round(solucion[1])} de cerveza Negra y {round(solucion[2])} de cerveza Baja")
    print(f"La ganancia total es {round(-res.fun)}")


#Ejercicio 5  (Sin resolver, ver como plantear las restricciones y función objetivo)
#   Funcion objetivo: Minimizar z = 68.3 * (x1+x2+x3+x4) + 69.5 * (x5+x6+x7+x8) + 71 * (x9,x10,x11,x12) + 71.2 * (x13+x14+x15+x16)
#  donde x1,x2,x3,x4 son las horas de trabajo del equipo 1 en las tareas M,N,P y Q respectivamente,
#  donde x5,x6,x7,x8 son las horas de trabajo del equipo 2 en las tareas M,N,P y Q respectivamente,
#  donde x9,x10,x11,x12 son las horas de trabajo del equipo 3 en las tareas M,N,P y Q respectivamente,
#  donde x13,x14,x15,x16 son las horas de trabajo del equipo 4 en las tareas M,N,P y Q respectivamente, 
#
#  Restricciones:
#      52 * x1 + 212 * x2 + 25 * x3 + 60 * x4 <= 220
#      57 * x5 + 218 * x6 + 23 * x7 + 57 * x8 <= 300
#      51 * x9 + 201 * x10 + 26 * x11 + 54 * x12 <= 245
#      56 * x13 + 223 * x14 + 21 * x15 + 55 * x16 <= 190
#   xi >= 0 para i q' pertenece a (0,16]


def ej5():
    c = np.array([68.3,68.3,68.3,68.3, 69.5,69.5,69.5,69.5, 71,71,71,71, 71.2,71.2,71.2,71.2], dtype="float")
    A_ub =  np.array([
        [52/349,212/349,25/349,60/349, 0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0, 57/355,218/355,23/355,57/355, 0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0, 51/332,201/332,26/332,54/332, 0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0, 56/355,223/355,21/355,55/355]
    ], dtype="float")
    b_ub = np.array([220,300,245,190], dtype="float")

    res = optimize.linprog(c, A_ub, b_ub)
    solucion = res.x

    print(res)
    x = np.reshape(res.x)
    print(np.round(x))



#Ejercicio 6


def ej6():
    costos = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB07/costos.dat')
    stock = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB07/stock.dat')
    demanda = np.loadtxt('Escritorio/LaboratorioAnnumerico/LAB07/demanda.dat')
    
    # x_{ij} -> y_k
    
    # x_1,1 -> y_1
    # x_1,2 -> y_2
    # ...
    # x_1,100 -> y_100
    # x_2,1 -> y_101
    # ...
    # x_100,100 -> y_10000

    c = costos.flatten()

    # b_ub
    b_ub = np.hstack([stock,-demanda])

    # A_ub
        # stock
        
        # sum_j x_{1,j} <= s_1
        # x_1,1 + x_1,2 + ... + x_1,100 <= s_1
        # y_1 + y_2 + ... + y_100 <= s_1

        # sum_j x_{2,j} <= s_2
        # x_2,1 + x_2,2 + ... + x_2,100 <= s_2
        # y_101 + y_102 + ... + y_200 <= s_2

        # A_s * y <= s
        # A_s.shape = (100,10000)

        #      1  2  ... 100 101 102 ... 200 201 ... 10000
        # 1    1  1  ...  1   0   0  ...  0   0  ...   0
        # 2    0  0  ...  0   1   1  ...  1   0  ...   0
        # ...
        # 100  0  0  ...  0   0   0  ...  0   0  ...   1

    A_s = np.zeros((100,10000))
    for i in range(100):
        A_s[i,(i*100):((i+1)*100)] = np.ones(100)

        # demanda

        # sum_i x_{i,1} >= d_1
        # x_1,1 + x_2,1 + ... + x_100,1 >= d_1
        # y_1 + y_101 + ... + y_9901 >= d_1

        # A_d * y >= d
        # A_d.shape = (100,10000)

        #      1  2  ... 100 101 102 ... 200 201 ... 10000
        # 1    1  0  ...  0   1   0  ...  0   1  ...   0
        # 2    0  1  ...  0   0   1  ...  0   0  ...   0
        # ...
        # 100  0  0  ...  1   0   0  ...  1   0  ...   1

    A_d = np.hstack([np.eye(100) for _ in range(100)])

    A_ub = np.vstack([A_s,-A_d])

    res = optimize.linprog(c, A_ub, b_ub)
    print(res)
    print(f"El costo mínimo es {res.fun}")

    x = np.reshape(res.x,(100,100))
    idx, idy = np.where(np.round(x))

    for i,j in zip(idx,idy):
        print(i,j,x[i,j])
