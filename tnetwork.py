import numpy as np
import pandas as pd
from trecfgnsga2_solver import TRecfGNSGA2 
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import circuit_utils as circuit 
import op_radialloadflow as radialflow

class TNetwork:

    def __init__(self):
        np.set_printoptions(precision=5, suppress=True)
        # Propiedades de base y escalado
        self.mvabase = 100  # Base de potencia en MVA
        self.kvnom = 1      # Tensión nominal en kV
        self.zbase = 1 / 100 # Impedancia base (inicial, se recalcula en getdata)

        # Datos de la red (leídos de archivos)
        self.graphs = None  # Estructura para almacenar grafos (matrices/arrays) - ej: perfiles de carga
        self.cost = None    # Costos asociados (vector/array) - ej: costo de pérdidas por MWh

        self.sources = None # Lista de fuentes de energía (array)
        self.branches = None# Lista de ramas del sistema (array)
        self.loads = None   # Lista de cargas (array)
        self.caps = None    # Lista de capacitores (array)

        # Relacionado con grafo/estructura (calculado en findlinks)
        self.N = 0 # Número total de nodos (sin virtuales)
        self.M = 0 # Número total de ramas (sin virtuales)
        self.NT = None # Número total de nodos (incluyendo virtuales si aplica). Calculado en findlinks.
        self.MT = None # Número total de ramas (incluyendo virtuales si aplica). Calculado en findlinks.
        # Propiedades de links y next (listas de adyacencia) - calculadas en findlinks
        self.link = [] # Lista de listas, link[i] contiene índices de ramas conectadas al nodo i+1
        self.next = [] # Lista de listas, next[i] contiene índices de nodos conectados al nodo i+1
        # Propiedades lnodes y ltypes (información combinada de ramas originales y virtuales) - calculadas en findlinks
        self.lnodes = np.array([]) # Nodos origen/destino (base 1) para todas las ramas (originales + virtuales)
        self.ltypes = np.array([]) # Tipo de interruptor para todas las ramas (originales + virtuales)

        # Estado de la red y resultados de la última evaluación
        # Usamos diccionarios en Python para simular estructuras MATLAB
        self.status = {'fun': np.zeros(5), 'opened': [], 'error': 0}
        # status0: Estado inicial de la red (antes de cualquier optimización)
        self.status0 = {'fun': np.zeros(5), 'opened': [], 'error': 0}

        # Configuración del flujo de carga
        self.time = 1        # Número de estados temporales (ej: perfiles de carga)
        self.lfmethod = 'radial2' # Método de flujo de carga ('radial2', 'radial1', 'inject')
        self.lffast = False # Indicador para usar un método rápido de flujo de carga (if available).
        self.lfmaxit = 100  # Número máximo de iteraciones permitidas para el flujo de carga.
        self.lftol = 1e-6    # Tolerancia para la convergencia del flujo de carga.

        # Indicador de mejora (utilizado por trecfgnsga2)
        self.isimprove= False

        # Variable para el índice de tiempo actual (pk en MATLAB) - utilizada por loadflowsparse
        self.pk = 0 # Inicializado a 0 (primer paso de tiempo)


    def exact(self, opened): # es equivalente a matLab
        self.lffast=False
        self.evaluate_off(opened)

    def closelink(self, close, ison, jfun): # revisar en movimiento
        """
        ison: numpy.ndarray (dtype=bool)
        close: int
        jfun: int
        """
        open_ = 0
        stat = None
        # Encuentra el lazo desde el nodo cerrado
        flag=ison.copy()
        loop = self.findloop(close, flag)
        # Filtra las ramas del lazo que son del tipo 'switch'
        opened = np.array([r for r in loop if self.ltypes[r-1]],dtype=int)-1

        # Se prueba el lazo con la rama 'close' forzada a activa
        isonj = ison.copy()
        isonj[close] = True

        ok, V,_ = self.loadflowsparse(isonj)
        if not ok:
            return open_, stat
        # Calcula corriente en cada rama 'opened'
        n = (self.branches[opened, 0:2].astype(int))-1
        Z = self.branches[opened, 4] + 1j * self.branches[opened, 5]
        I = np.abs((V[n[:, 0]] - V[n[:, 1]]) / Z)
        # Ordena por menor corriente
        I = np.round(I, 4)
        index = np.argsort(I,  kind='mergesort')
        minfk = float("inf")
        lastloss = float("inf")
       
        for j in index:
            isonj[opened] = True
            isonj[opened[j]] = False
            self.evaluate_on(isonj)
            if self.status["error"] == 0:
                statj = self.status.copy()
                lossj = statj["fun"][jfun]
                if lossj > lastloss:
                    break
                lastloss = lossj
                if lossj < minfk:
                    open_ = opened[j]
                    stat = statj.copy()
            
        return open_, stat


    def SLE_Search(self, stat, jfun):
        # Estado inicial
        if stat is None:
            stat = self.status0

        # Convertir 'opened' en un array 1D
        openset = stat["opened"].flatten()
        k = 0

        while k < len(openset):
            # Crear vector ison de interruptores encendidos
            ison = np.ones(self.M, dtype=bool)
            ison[stat["opened"]] = False
            # Ejecutar la función closelink
            openk, statk = self.closelink(openset[k], ison, jfun)
            # Verificar mejora y agregar nuevos interruptores abierto

            if openk and statk["fun"][jfun] < stat["fun"][jfun]:
                openk = np.atleast_1d(openk)
                openset = np.concatenate((openset, openk))
                stat = statk

            k += 1
        self.status = stat



    def findloop(self, closed, flag):
        """
        close: int index
        flag: numpy.ndarray (dtype=bool)
        """
        if self.M<self.MT:
                flag = np.concatenate((flag, np.ones((self.MT - self.M, 1),dtype=bool)), axis=0)
        flag[closed]=False
        mark=np.zeros(len(self.next),dtype=bool)
        def search(n, flag, mark):
                ok = False
                loop = []
                stack = [(n[0], [])]

                while stack:
                    nodo_actual, camino = stack.pop()

                    if mark[nodo_actual - 1]:
                          continue
                    mark[nodo_actual - 1] = True

                    if nodo_actual == n[1]:
                          ok = True
                          loop = camino
                          break

                    for lnk_idx, vecino in enumerate(self.next[nodo_actual - 1]):
                          rama = self.link[nodo_actual - 1][lnk_idx]
                          if flag[rama - 1]:
                                flag[rama - 1] = False
                                stack.append((vecino, [rama] + camino))

                return ok, loop


        ok,loop=search(self.lnodes[closed,:],flag,mark)
        if not ok:
                raise Exception("No hay lazo")
        return loop

    
    def loadflowsparse(self, ison): # es equivalente a MatLab
        n = self.branches[np.where(ison)][:, [0, 1, 4, 5]]
        Z=n[:,2]+1j*n[:,3]
        nl=(self.loads[:,0].astype(int))-1
        Slp=self.loads[:,1]+1j*self.loads[:,2]
        Sl=Slp.copy()
        if self.caps!=None:
                nc= (self.caps[:,0].astype(int))-1
                Yc= 1j*self.caps[:,1]
        s=(self.sources[:,0].astype(int))-1
        V= np.ones((self.N,1),dtype=complex) #vector columna de 1
        t=self.pk
        #formacion de Ybus
        Ybus = sp.csc_matrix((self.N, self.N), dtype=np.complex128)
        Ybus=Ybus.tolil()
        for i in range(n.shape[0]):
                from_node, to_node = int(n[i, 0])-1, int(n[i, 1])-1
                yi = 1 / Z[i]

                Ybus[from_node, from_node] += yi
                Ybus[to_node, to_node] += yi
                Ybus[from_node, to_node] -= yi
                Ybus[to_node, from_node] -= yi
        Ybus = Ybus.tocsc()

        V[s]=self.sources[:,2]
        if self.time>1:
                Sl = (Slp * self.graphs[t, (self.loads[:, 3].astype(int)) - 1]).reshape(-1,1)

        NB= np.zeros(self.N, dtype=bool)
        NB[s]=True
        N1=s
        N2=np.where(~NB)[0]

        #fem debida a las fuentes
        Ybus_N2_N2 = Ybus[N2, :][:, N2]

        lu=splu(Ybus_N2_N2)
        I21 = Ybus[N2, :][:, N1] @ V[N1] #multiplicacion de matrices
        FEM = -lu.solve(I21)
        V[N2]=FEM
        eps=1
        it=0
        tol=self.lftol
        maxit=self.lfmaxit
        while (it<maxit) and (eps>tol):
                it+=1
                V0=V.copy()
                I= np.zeros((self.N,1), dtype=complex)
                I[nl]-=np.conj(Sl/V[nl])
                if self.caps!=None:
                     I[nc]-=V[nc]*Yc
                V[N2]=FEM+ lu.solve(I[N2])
                eps= np.max(np.abs(V-V0))
        ok = eps <= tol
        I[N1]=Ybus[N1,:]@ V
        Sg =V[s]*np.conj(I[s])
        dS=np.sum(Sg)-np.sum(Sl)
        return ok, V.flatten(), dS

    def prim(self): #revisar despues
        # CREA RADIAL ALEATORIO
        N1 = []
        N2 = []

        # ramas no analizadas
        flag = np.ones(self.MT, dtype=complex)

        # nro de ramas conectadas
        M1 = self.N - 1

        # nodo origen aleatorio
        # D = 1 + round(np.random.rand(1) * (self.N - 1))
        # vamos a poner la fuente siempre
        S = self.sources.shape[0]

        if S > 1:
                D = [self.N] + self.sources[:, 0].tolist()
                N1= list(range(self.MT - S + 1, self.MT + 1))
                N1.extend(N1)
                for idx in N1:
                     flag[idx - 1] = 0
        else:
                D = self.sources[0]


        while len(N1) < M1:
                # conjunto de posibles ramas
                R = []
                n = []
                for i_node in D:

                    if 0 <= i_node - 1 < len(self.link):
                          for j in range(len(self.link[i_node - 1])):
                                r = self.link[i_node - 1][j]
                                if flag[r - 1]:
                                     R.append(r)
                                     n.append(self.next[i_node - 1][j])

                if len(R)==0:
                    break

                # rama a conectar
                i = np.random.randint(len(R))
                r = R[i]

                # saca rama r de las posibles
                flag[r - 1] = 0

                if n[i] not in D:
                    # es radial
                    D.append(n[i])
                    N1.append(r)
                else:
                    N2.append(r)

        # si quedan ramas no utilizadas se ponen en N2
        u = np.where(flag == 1)[0] + 1
        if u.size > 0:
                N2.extend(u.tolist())

        # N2 = sorted(N2) # Uncomment if you need N2 sorted

        return N1, N2


    def improve(self, y1, jfun): # es equivalente a matLab
        if self.isimprove:
                self.evaluate_off(y1)
                self.SLE_Search(self.status,jfun)
                y1=self.status["opened"]
                return y1


    def uniformcrossover(self, p1, p2, u): #probar mientras corre
        L = self.MT - self.NT + 1
        p1 = p1[:L].copy()
        p2 = p2[:L]
        ison = np.ones(self.MT, dtype=bool)
        ison[p1] = False

        for k in u:
                loop = self.findloop(p1[k], ison)
                idx = np.flatnonzero(loop == p2[k])
                if idx.size:
                     j = idx[0]
                else:
                     loop = loop[[0, -1]]
                     j = np.random.randint(loop.size)

                ison[p1[k]] = True
                ison[loop[j]] = False
                p1[k] = loop[j]

        return p1


    def mutation(self, p1, sigma): #probar mientras corre
        L=self.MT-self.NT+1
        p1=p1[:L].copy()
        r = np.random.randint(1, int(np.ceil(sigma * L)) + 1)
        u = np.random.choice(L, size=r, replace=False)
        ison = np.ones(self.MT, dtype=bool)
        ison[p1] = False
        for k in u:

                loop = self.findloop(p1[k], ison)
                loop = [loop[0], loop[-1]]
                j = np.random.randint(len(loop))

                ison[p1[k]]   = True
                ison[loop[j]] = False
                p1[k]         = loop[j]

        return p1

 
    def getdata(self, filename): #arreglado
        """
        Lee la informacion de la red desde un archivo xlsx
        """
        try:

            cost_df = pd.read_excel('exgraphs.xlsx', sheet_name='Cost')
            self.cost = cost_df.values.flatten() * 1000
            graphs_df = pd.read_excel('exgraphs.xlsx', sheet_name='Graph')
            self.graphs = graphs_df.values
            #obtenemos los nombres de las hojas en el xlsx
            xls = pd.ExcelFile(filename)
            desc = xls.sheet_names

            if 'Sources' not in desc:
                raise ValueError('falta la hoja Sources en el archivo xlsx')

            sources_df = pd.read_excel(filename, sheet_name='Sources')
            self.sources = sources_df.values
            self.kvnom = self.sources[0, 1]

            self.zbase = self.kvnom**2 / self.mvabase


            if 'Branches' not in desc:
                raise ValueError('falta la hoja Branches en el archivo xlsx')

            branches_df = pd.read_excel(filename, sheet_name='Branches')
            self.branches = branches_df.values
            self.branches[:, 4:6] = self.branches[:, 4:6] / self.zbase
            switches = np.nonzero(self.branches[:, 3])[0]
            if np.size(switches)==0:
                raise Exception("no hay interruptores")
            if 'Loads' not in desc:
                raise ValueError('falta la hoja Loads en el archivo xlsx')

            loads_df = pd.read_excel(filename, sheet_name='Loads')
            self.loads = loads_df.values
            self.loads[:, 1:3] = self.loads[:, 1:3] / self.mvabase

            if 'Caps' in desc:
                caps_df = pd.read_excel(filename, sheet_name='Caps')
                self.caps = caps_df.values
                self.caps[:, 1] = self.caps[:, 1] / self.mvabase

            n= self.branches[:, 0:2]
            self.N = int(np.max(n))

            self.findlinks()

        except FileNotFoundError:
            raise FileNotFoundError(f"el archivo '{filename}' o 'exgraphs.xlsx' no se encuentra.")

    
    def loadflow(self, ison): # es equivalente a matlab

        n = self.branches[np.where(ison)][:, [0, 1, 4, 5]]
        Z = n[:, 2] + 1j * n[:, 3]
        nl=(self.loads[:,0].astype(int))-1

        Slp = self.loads[:, 1] + 1j * self.loads[:, 2]
        Sl=Slp.copy()

        if self.caps is not None:
                nc = self.caps[:, 0].astype(int) - 1 #ajustamos los indices de base 1 a base 0
                Yc = 1j * self.caps[:, 1]
        s = (self.sources[:, 0].astype(int)) - 1
        #formacion de Ybus
        # Crea una matriz dispersa de ceros
        Ybus = sp.csc_matrix((self.N, self.N), dtype=np.complex128)

        for i in range(n.shape[0]):
                from_node, to_node = int(n[i, 0])-1, int(n[i, 1])-1
                yi = 1 / Z[i]

                Ybus[from_node, from_node] += yi
                Ybus[to_node, to_node] += yi
                Ybus[from_node, to_node] -= yi
                Ybus[to_node, from_node] -= yi
        NB = np.zeros(self.N, dtype=bool)
        NB[s] = True
        N1 = s
        N2 = np.where(~NB)[0]

        Ybus_N2_N2 = Ybus[N2, :][:, N2]
        lu=splu(Ybus_N2_N2)

        V=np.ones((self.N,self.time),dtype=complex)

        loss=np.zeros(self.time)
        pgen=np.zeros(self.time)

        tol=self.lftol
        maxit=self.lfmaxit
        for t in range(self.time):
                V[s,t]=self.sources[:,2]

                if self.time>1:
                    Sl = Slp * self.graphs[t, self.loads[:, 3].astype(int) - 1]
                # FEM debida a las fuentes
                I21 = Ybus[N2, :][:, N1] @ V[N1, t] #multiplicacion de matrices
                FEM = -lu.solve(I21)
                V[N2, t] = FEM
                eps  = 1
                it   = 0
                while eps > tol and it < maxit:
                    it+=1
                    V0=V[:,t].copy()
                    I = np.zeros(self.N, dtype=np.complex128)
                    I[nl] = I[nl] - np.conj(Sl / V[nl, t])

                    if self.caps is not None and self.caps.size>0:
                          I[nc] -= V[nc, t] * Yc
                    V[N2, t] = FEM + lu.solve(I[N2])
                    eps = np.max(np.abs(V[:, t] - V0))
                ok = eps <= tol
                if not ok:
                    return ok, V, loss, pgen

                I[N1] = Ybus[N1, :] @ V[:, t]
                Sg = V[s, t] * np.conj(I[s])
                dS = np.sum(Sg) - np.sum(Sl)

                loss[t] = np.real(dS)
                pgen[t] = np.real(Sg).sum()

        return ok, V, loss, pgen


    def optimalordering(self, ison=None):
        s = self.sources[:, 0].astype(np.int32)

        if ison is None:
            ison = self.branches[:, 2].astype(bool)

        opened = np.where(~ison)[0]

        # n puede tener columnas mixtas, así que usamos float64
        n = self.branches[ison][:, [0, 1, 4, 5]].astype(np.float64)
        M1 = n.shape[0]

        order = np.zeros(M1, dtype=np.int32)
        circ = np.zeros((self.N, 1), dtype=np.int32)
        m_counter = [0] # Inicializa m como una lista con un solo elemento
        for k in range(1, len(s) + 1):
                flag = np.ones(M1, dtype=np.uint8)
                ok= circuit.findcircuit(n, circ, order, flag, s[k - 1], k,m_counter)

                if not ok:
                     return False, n, opened
                # Recolectar todos los índices asignados en order > 0
        recorridos = np.where(order > 0)[0]

        # Preservar el orden según el valor de 'order'
        orden_valores = order[recorridos]
        sorted_indices = recorridos[np.argsort(orden_valores)]

        # Reordenar 'n' finalmente
        n = n[sorted_indices]
        if self.caps is not None:
                ok = (
                     np.all(circ[self.loads[:, 0].astype(int)] > 0)
                     and np.all(circ[self.caps[:, 0]] > 0)
                )
        else:
                ok = np.all(circ[self.loads[:, 0].astype(int)] > 0)

        return ok, n, opened

    def evaluate_off(self, isopen):
        ison=None
        if isopen is None:
            ison=np.ones(self.M)
            ison[isopen]=0
        self.evaluate_on(ison)
    
    def radialloadflow(self, ison):
        ok,n,_=self.optimalordering(ison)
        N = int(self.N)
        T = int(self.time)
        return radialflow.radialloadflow(ok, n, self.sources, self.loads, self.graphs, self.caps, N, T,self.lftol,self.lfmaxit,self.lfmethod) #esta funcion esta acelerada en c
    
    def evaluate_on(self, ison):
        isempty=ison is None
        if isempty:
            ison=self.branches[:,2].astype(bool)
        opened= np.where(~ison)[0]
        err=None
        if self.lfmethod == 'inject':
            ok, V, loss, pgen=self.loadflow(ison)
        else:
            if not self.lffast:
                ok, err, V, loss, pgen, smin = self.radialloadflow(ison)
        self.status['error'] = err
        self.status['fun'] = np.zeros(5)
        self.status["opened"]=opened

        if not ok:
                if self.status['error'] == 1:
                    self.status['fun'][:] = 20000

                elif self.status['error'] == 2:
                    self.status['fun'][:] = 10000
                return
        if isempty:
                self.pk=np.argmax(pgen)

        MW = np.max(loss)* self.mvabase
        MWh = np.sum(loss)* self.mvabase
        Cost=np.sum(loss * self.cost)* self.mvabase
        if self.time==1:
                Cost=MWh
        Vmin = np.min(np.abs(V))
        Smin = np.min(smin)

        self.status['fun'][0] = MW 
        self.status['fun'][1] = MWh 
        self.status['fun'][2] = Cost 
        self.status['fun'][3] = -Vmin
        self.status['fun'][4] = -Smin

        if isempty:
            self.status0 = self.status.copy()

   
    def findlinks(self): #es equivalente a matLab

        n = self.branches[:, 0:2].astype(int)
        t = self.branches[:, 3].astype(int)
        self.M = n.shape[0]
        self.NT = self.N
        self.MT = self.M
        S = self.sources.shape[0]

        if S > 1:
                self.NT = self.N + 1
                self.MT = self.M + S
                new_rows=np.column_stack((self.NT*np.ones(S),self.sources[:,0]))
                n=np.concatenate((n,new_rows))
                t=np.concatenate((t,np.zeros(S)))

        self.lnodes = n
        self.ltypes = t
        self.next = [[] for _ in range(self.NT)]
        self.link = [[] for _ in range(self.NT)]

        for i in range(self.MT):

                self.next[n[i, 0] - 1].append(n[i, 1])
                self.link[n[i, 0] - 1].append(i + 1)

                self.next[n[i, 1] - 1].append(n[i, 0])
                self.link[n[i, 1] - 1].append(i + 1)

    def nsga2solver1(self, npop: int, pop=None):
        """
        Puerto directo del wrapper MATLAB que:
            1. Crea un solver NSGA-II especializado (`TRecfGNSGA2`)
            2. Ejecuta la optimización
            3. Para cada objetivo, toma el individuo con coste mínimo
               y lo evalúa de forma “exacta” en la red que vive en `self`.
            4. Devuelve `self` (por fluidez) y el objeto solver
        """
        # 1) Instanciar el solver hijo
        solver = TRecfGNSGA2(self, npop, pop)
        solver.verb = True
        # 2) Lanzar NSGA-II
        solver.nsga2()                          # ← implementado en la super-clase

        # 3) Post-proceso: mejor individuo por cada objetivo
        for i in range(solver.nobjs):
            j = np.argmin(solver.costfunc[:, i])
            opened = solver.variable[j, :]      # cromosoma ganador
            self.exact(opened)                  # evaluación fina
            self.printstatus()                  # log

        # 4) Exponer matrices finales (opcional)
        variable = solver.variable
        costfunc = solver.costfunc
        parfunc  = solver.parfunc

        # En Python solemos devolver todo en un tuple o dict
        return self, solver, variable, costfunc, parfunc


    def printstatus(self):
        print("Opened =", end=" ")
        print(*self.status["opened"])  # Imprime los valores de 'opened' separados por espacio
        print("\nLosses = {:.6f} MW".format(self.status["fun"][0]))
        print("Losses = {:.6f} MWh".format(self.status["fun"][1]))
        print("Cost   = {:.6f} $".format(self.status["fun"][2]))
        print("Vmin   = {:.6f} pu".format(-self.status["fun"][3]))
        print("SImin  = {:.6f} pu".format(-self.status["fun"][4]))
        print("\n")
