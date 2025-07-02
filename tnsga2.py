import numpy as np
import random
import matplotlib.pyplot as plt

class TNSGA2:
   
    def __init__(self):
    
        self.npop = 100       
        self.ngen = 100       
        self.nvars = 0        
        self.nparm = 0        
        self.nobjs = 0       
        self.pcross = 0.8     
        self.maxit1 = 10      
        self.maxit2 = 50     
        self.sigma = 0.1      
        self.crossf = 0.5     
        self.verb = True      
        self.bestfun = []     
        self.objstr = []      
        self.xlim = []        
        self.variable = []    
        self.costfunc = []    
        self.parfunc = []     

        # Atributo para simular la variable global 'objfun' de MATLAB
        self.objfun = [] # Almacenará los mejores valores de objetivo por generación

        # Propiedad para simular tic/toc
        self._start_time = None

    def evaluate_objective(self, x):
        #esto es un placeholder
       
        return np.array([]), np.array([])


    def dispvte(self, i):
        # otro placeholder?
        pass

    def generate(self):
        xmin = self.xlim[0, :]
        xmax = self.xlim[1, :] 
        xinc = self.xlim[2, :] 

        V = self.nvars 
        f = np.zeros(V)

        for j in range(V):
            f[j] = xmin[j] + (xmax[j] - xmin[j]) * random.random() 
            if xinc[j] != 0:
                # Redondear al múltiplo más cercano del paso absoluto
                f[j] = round(f[j] / abs(xinc[j])) * abs(xinc[j])

        f = np.clip(f, xmin, xmax)

        return f


    def nsga2(self):

        pop = self.npop 
        gen = self.ngen 
        V = self.nvars 
        P = self.nparm 
        M = self.nobjs 

        chromosome = self.initialize_variables(2 * pop)
        chromosome = self.non_domination_sort_mod(chromosome)
        chromosome = self.replace_chromosome(chromosome, pop)

        flast = np.full(M, np.inf) 
        it = 0 
        self.objfun = np.zeros((gen, M)) 

        for i in range(gen): 
    
            pool = round(pop / 2) 
            tour = 2 
            parent_chromosome = self.tournament_selection(chromosome, pool, tour)
            offspring_chromosome = self.genetic_operator(parent_chromosome)
            intermediate_chromosome = np.vstack((chromosome[:, :(V + P + M)], offspring_chromosome[:, :(V + P + M)]))

            intermediate_chromosome = self.non_domination_sort_mod(intermediate_chromosome)

            chromosome = self.replace_chromosome(intermediate_chromosome, pop)

            rank = chromosome[:, -2] 
            rank1 = np.sum(rank == 1)
            fmin = np.min(chromosome[:, V : V + M], axis=0) 

            self.objfun[i, :] = fmin
            self.bestfun = fmin.copy() 

            it += 1
            if np.any(fmin < flast): 
                it = 0 

            flast = fmin.copy() 

            stop_condition1 = (it >= self.maxit1) and (rank1 == pop) 
            stop_condition2 = (it >= self.maxit2) and (rank1 > 0.6 * pop) 

            if stop_condition1 or stop_condition2:
                break 

        self.variable = chromosome[:, :V] 
        self.costfunc = chromosome[:, V : V + M] 
        self.parfunc = chromosome[:, V + M : V + M + P] 



    def initialize_variables(self, N):
        
        V = self.nvars 
        P = self.nparm 
        M = self.nobjs 
        f = np.zeros((N, V + P + M))

        if self.variable.shape[0] > 0: 
            num_existing = self.variable.shape[0]
            if self.variable.shape[1] >= V:
    
                cols_to_copy = min(self.variable.shape[1], V + M + P)
                f[:num_existing, :cols_to_copy] = self.variable[:num_existing, :cols_to_copy]
            else:
                num_existing = 0 

        else:
             num_existing = 0 
       
        for i in range(num_existing, N):
            variables = self.generate()

            fobj, fpar = self.evaluate_objective(variables) 

            if fobj is None or fobj.size != M:
                fobj = np.full(M, 1e9) # Asignar penalización alta

            if fpar is None or fpar.size != P:
                fpar = np.zeros(P) # Usar ceros si el tamaño no coincide

            # Almacenar variables, objetivos y parámetros en la matriz de población
            f[i, :V] = variables
            f[i, V : V + M] = fobj
            f[i, V + M : V + M + P] = fpar

        return f


    

    def non_domination_sort_mod(self, x: np.ndarray) -> np.ndarray:
        """
        Réplica directa del MATLAB `non_domination_sort_mod`.
        `x` se modifica IN-PLACE igual que en el original.

        Dimensiones en `x`   (todas 0-based):
        ┌─[0 … V-1]                 → variables de decisión
        ├─[V … V+M-1]               → objetivos
        ├─[V+M … V+M+P-1]           → parámetros (si los hubiera)
        ├─[V+M+P]                   → columna de RANK
        ├─[V+M+P+1 … V+M+P+M]       → distancias de crowding (una por objetivo)
        └─[V+M+P+M+1]               → suma de distancias (crowding total)
        """
        N   = x.shape[0]
        V   = self.nvars
        P   = self.nparm
        M   = self.nobjs

        front = 1                                            # Contador de frentes (1-based)
        F = {front: []}                                     # Diccionario de frentes
        individual = [dict(n=0, p=[]) for _ in range(N)]    # Estructura auxiliar

        # ----------------- 1. Dominancia y primer frente -----------------
        for i in range(N):
            for j in range(N):
                dom_less  = dom_equal = dom_more = 0
                for k in range(M):
                    if   x[i, V+k] <  x[j, V+k]: dom_less  += 1
                    elif x[i, V+k] == x[j, V+k]: dom_equal += 1
                    else:                        dom_more  += 1

                if dom_less == 0 and dom_equal != M:
                    individual[i]['n'] += 1
                elif dom_more == 0 and dom_equal != M:
                    individual[i]['p'].append(j)

            if individual[i]['n'] == 0:                    # No está dominado
                x[i, V+M+P] = 1                            # RANK = 1   (columna V+M+P)
                F[front].append(i)

        # ----------------- 2. Encontrar frentes sucesivos ----------------
        while F.get(front):
            Q = []
            for i_idx in F[front]:
                for j in individual[i_idx]['p']:
                    individual[j]['n'] -= 1
                    if individual[j]['n'] == 0:
                        x[j, V+M+P] = front + 1            # Asignar nuevo RANK
                        Q.append(j)
            front += 1
            F[front] = Q

        # ----------------- 3. Ordenar por frente -------------------------
        idx_fronts = np.argsort(x[:, V+M+P])
        sorted_by_front = x[idx_fronts, :]

        # ----------------- 4. Distancia de crowding por frente -----------
        z = sorted_by_front.copy()                         # Salida final
        current_index = -1
        for fr in range(1, front):                         # front ya está +1
            ids = F[fr]
            if not ids:                                   # Frente vacío
                continue

            prev = current_index + 1
            current_index += len(ids)
            y = sorted_by_front[prev:current_index+1, :]   # Individuos de este frente

            # Para cada objetivo…
            for m in range(M):
                obj_col = V + m
                sort_idx = np.argsort(y[:, obj_col])
                f_max, f_min = y[sort_idx[-1], obj_col], y[sort_idx[0], obj_col]

                # Bordes → ∞
                y[sort_idx[0],  V+M+P+1+m] = np.inf
                y[sort_idx[-1], V+M+P+1+m] = np.inf

                # Interiores
                for pos in range(1, len(sort_idx)-1):
                    nxt = y[sort_idx[pos+1], obj_col]
                    prv = y[sort_idx[pos-1], obj_col]
                    if f_max == f_min:
                        y[sort_idx[pos], V+M+P+1+m] = np.inf
                    else:
                        y[sort_idx[pos], V+M+P+1+m] = (nxt - prv) / (f_max - f_min)

            # Sumar distancias parciales
            crowd_cols = slice(V+M+P+1, V+M+P+1+M)         # M columnas de crowding
            y[:, V+M+P+M+1] = np.sum(y[:, crowd_cols], axis=1)

            # Recortar / copiar de vuelta
            z[prev:current_index+1, :] = y

        return z
   

    def tournament_selection(self,
                            chromosome: np.ndarray,
                            pool_size: int,
                            tour_size: int) -> np.ndarray:
        """
        Réplica directa del `tournament_selection` de MATLAB.

        chromosome : ndarray (N × C)      Población actual
        pool_size  : int                  Tamaño del mating-pool a devolver
        tour_size  : int                  Tamaño del torneo

        Convención de columnas (0-based):
            C-2  → RANK        (penúltima)
            C-1  → DISTANCIA   (última)
        La función NO altera `chromosome`.
        """
        pop, variables = chromosome.shape
        rank_col     = variables - 2          # penúltima
        distance_col = variables - 1          # última

        f = np.empty((pool_size, variables), dtype=chromosome.dtype)

        rng = np.random.default_rng()         # generador moderno

        for i in range(pool_size):

            # -------- 1. Elegir candidatos ÚNICOS para el torneo ----------
            # (replicamos el while/round original con un bucle explícito)
            candidate = np.empty(tour_size, dtype=int)
            for j in range(tour_size):
                idx = rng.integers(0, pop)    # 0 … pop-1
                while (idx in candidate[:j]): # evitar duplicados
                    idx = rng.integers(0, pop)
                candidate[j] = idx

            # -------- 2. Extraer rank & distance de cada candidato --------
            c_obj_rank     = chromosome[candidate, rank_col]
            c_obj_distance = chromosome[candidate, distance_col]

            # -------- 3. Resolución del torneo (misma lógica que MATLAB) --
            min_rank = c_obj_rank.min()
            ties     = np.where(c_obj_rank == min_rank)[0]     # índices en `candidate`

            if len(ties) > 1:
                # desempate por mayor distancia de crowding
                tie_dist = c_obj_distance[ties]
                best_idx_within_ties = ties[np.argmax(tie_dist)]
                winner = candidate[best_idx_within_ties]
            else:
                winner = candidate[ties[0]]

            # -------- 4. Copiar el individuo ganador al mating pool -------
            f[i, :] = chromosome[winner, :]

        return f

    def genetic_operator(self, parent_chromosome: np.ndarray) -> np.ndarray:
        """
        Réplica 1-a-1 del `genetic_operator` de MATLAB.

        parent_chromosome : ndarray (N × (V+M+P))   Población padre
        Devuelve           : ndarray (⚠ tamaño variable)  Hijos generados
        ────────────────────────────────────────────────────────────────────
        Convención columnas (0-based):
            0 … V-1                  → variables de decisión
            V … V+M-1                → objetivos
            V+M … V+M+P-1            → parámetros adicionales
            (rank / crowding no se rellenan aquí; se añaden luego)
        """
        N = parent_chromosome.shape[0]
        V = self.nvars
        P = self.nparm
        M = self.nobjs

        rng = np.random.default_rng()
        children = []                           # lista dinámica

        was_crossover = False
        was_mutation  = False
        p = 0                                    # índice hijo (solo para reflejar MATLAB)

        for _ in range(N):

            if rng.random() < self.pcross:      # --- Crossover --------------------
                # Elegir dos padres distintos
                parent1_idx = rng.integers(0, N)
                parent2_idx = rng.integers(0, N)
                tries = 0
                while np.array_equal(parent_chromosome[parent1_idx],
                                    parent_chromosome[parent2_idx]):
                    parent2_idx = rng.integers(0, N)
                    tries += 1
                    if tries > 2 * N:
                        break

                parent1 = parent_chromosome[parent1_idx]
                parent2 = parent_chromosome[parent2_idx]

                # Crossover sobre las V variables
                child1_dec, child2_dec = self.crossover(parent1[:V], parent2[:V])

                # Evaluar objetivos y parámetros
                fobj1, fpar1 = self.evaluate_objective(child1_dec)
                fobj2, fpar2 = self.evaluate_objective(child2_dec)

                # Construir cromosomas hijos
                child1 = np.empty(V + M + P, dtype=parent_chromosome.dtype)
                child2 = np.empty_like(child1)
                child1[:V]               = child1_dec
                child1[V:V+M+P]          = np.hstack((fobj1, fpar1))
                child2[:V]               = child2_dec
                child2[V:V+M+P]          = np.hstack((fobj2, fpar2))

                was_crossover = True
                was_mutation  = False

            else:                               # --- Mutación ---------------------
                parent3_idx = rng.integers(0, N)
                child3      = parent_chromosome[parent3_idx].copy()

                # Mutar solo las variables
                child3[:V] = self.mutate(child3)

                # Reevaluar objetivos y parámetros
                fobj3, fpar3 = self.evaluate_objective(child3[:V])
                child3[V:V+M+P] = np.hstack((fobj3, fpar3))

                was_mutation  = True
                was_crossover = False

            # --- Copiar a la lista de salida, manteniendo la lógica original ----
            if was_crossover:
                children.append(child1)
                children.append(child2)
                was_crossover = False
                p += 2                         # (solo por simetría con MATLAB)

            elif was_mutation:
                children.append(child3[:V+M+P])  # igual que MATLAB: sin rank/crowding
                was_mutation = False
                p += 1

        return np.vstack(children) if children else np.empty((0, V+M+P))
    

    def crossover(self, p1: np.ndarray, p2: np.ndarray):
        """
        Swap-crossover idéntico al MATLAB original.

        p1, p2 : ndarrays de longitud V (variables de decisión)
        Devuelve
            y1, y2 : hijos (clonados y cruzados + `self.update`)

        Algoritmo:
        1. Seleccionar aleatoriamente `round(V * crossf)` posiciones SIN reemplazo.
        2. Intercambiar los valores de esas posiciones entre p1 y p2.
        3. Pasar cada hijo por `self.update()` para forzar factibilidad.
        """
        V = self.nvars

        rng = np.random.default_rng()
        k   = round(V * self.crossf)                 # nº de posiciones a intercambiar
        u   = rng.choice(V, size=k, replace=False)   # índices 0 … V-1

        # Copias de los padres
        y1 = p1.copy()
        y2 = p2.copy()

        # Intercambio
        y1[u] = p2[u]
        y2[u] = p1[u]

        # Ajustes post-crossover (por ejemplo, clipping o reparaciones)
        y1 = self.update(y1)
        y2 = self.update(y2)

        return y1, y2

    def mutate(self, p: np.ndarray) -> np.ndarray:
        """
        Mutación gaussiana idéntica a la versión MATLAB.

        p : ndarray (≥ V)   Cromosoma padre (se usa solo la parte de variables)
        Devuelve
            y : ndarray (V)  Variables mutadas y reparadas con `self.update`.
        """
        V = self.nvars
        x = p[:V].copy()                                   # copia de las variables

        # Nº de genes a mutar
        n_mu = int(np.ceil(self.mu * V))

        # Índices únicos a mutar (0 … V-1)
        rng = np.random.default_rng()
        j   = rng.choice(V, size=n_mu, replace=False)

        # Perturbación gaussiana
        x[j] += self.sigma * rng.standard_normal(size=n_mu)

        # Reparación / clipping personalizado
        y = self.update(x)
        return y

    def replace_chromosome(self,
                        intermediate_chromosome: np.ndarray,
                        pop: int) -> np.ndarray:
        """
        Selecciona la nueva población de tamaño `pop` a partir de
        `intermediate_chromosome` siguiendo el mismo esquema
        (rank → crowding‐distance) que en el MATLAB original.

        Columnas (0-based):
            0 … V-1                   variables
            V … V+M-1                 objetivos
            V+M … V+M+P-1             parámetros
            V+M+P                     rank
            V+M+P+1                   crowding distance
        """
        N, _ = intermediate_chromosome.shape
        V, P, M = self.nvars, self.nparm, self.nobjs
        rank_col     = V + M + P
        distance_col = rank_col + 1          # crowding distance

        # 1) Ordenar toda la población por RANK ascendente
        idx_rank  = np.argsort(intermediate_chromosome[:, rank_col])
        sorted_ch = intermediate_chromosome[idx_rank]

        max_rank = int(sorted_ch[:, rank_col].max())

        selected = []                        # lista dinámica de filas

        for r in range(1, max_rank + 1):
            # a) Individuos del frente r
            in_front = np.where(sorted_ch[:, rank_col] == r)[0]
            if in_front.size == 0:
                continue

            front_rows = sorted_ch[in_front]
            room_left  = pop - len(selected)

            # b) ¿Cabe todo el frente?
            if front_rows.shape[0] <= room_left:
                selected.extend(front_rows)
                if len(selected) == pop:     # cupo exacto
                    break
            else:
                # c) Desempate por distancia (descendente)
                order = np.argsort(front_rows[:, distance_col])[::-1]
                selected.extend(front_rows[order][:room_left])
                break

        return np.vstack(selected)

    def update(self, x: np.ndarray) -> np.ndarray:
        """
        Ajusta el vector `x` a:
        • rangos   [vmin, vmax]
        • múltiplos de un incremento (si existe)  xlim[2, :]
        
        self.xlim debe ser un array 3×V:
            fila 0 → vmin
            fila 1 → vmax
            fila 2 → incremento (0 ⇒ continuo, >0 ⇒ discreto)
        """
        vmin   = self.xlim[0]          # (V,)
        vmax   = self.xlim[1]          # (V,)
        inc    = self.xlim[2]          # (V,)

        mask   = inc != 0              # True donde hay step discreto
        if np.any(mask):
            x = x.copy()               # evitamos sobre-escribir argument in-place
            x[mask] = np.round(x[mask] / inc[mask]) * inc[mask]

        # Clip a los límites
        x = np.minimum(x, vmax)
        x = np.maximum(x, vmin)
        return x

    def issorted(self,p1: np.ndarray,
                p2: np.ndarray,
                order: np.ndarray | list,
                i: int = 0) -> int:
        """
        Devuelve 1 si el vector `p1` es lexicográficamente ≤ `p2`
        siguiendo el orden dado; 0 en caso contrario.
        
        order : secuencia de índices (0-based) que define la prioridad
        i     : posición actual en la recursión (no tocar al llamar)
        """
        if i >= len(order):
            return 1                                      # agotó el orden → iguales
        idx = order[i]
        if p1[idx] < p2[idx]:
            return 1
        elif p1[idx] == p2[idx]:
            return self.issorted(p1, p2, order, i + 1)
        else:
            return 0
