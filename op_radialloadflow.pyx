# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=True

import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

ctypedef np.float64_t FLOAT64
ctypedef np.complex128_t COMPLEX64
ctypedef np.int32_t INT32

@cython.cdivision(True)
def radialloadflow(
    bint ok,
    np.ndarray[FLOAT64, ndim=2] n,
    np.ndarray[FLOAT64, ndim=2] sources,
    np.ndarray[FLOAT64, ndim=2] loads,
    np.ndarray[FLOAT64, ndim=2] graphs,
    object caps,
    int N,
    int time,
    double lftol,
    int lfmaxit,
    str lfmethod
):
    if not ok:
        return False, 2, None, None, None, None

    cdef:
        # Declaración de enteros y otros tipos primitivos
        int M1 = n.shape[0]
        int t, i, it
        double eps, SI, V1abs, aux,tmp
        bint caps_present = caps is not None

        # Declaración y asignación explícita de arrays de NumPy
        np.ndarray[INT32, ndim=1] s = np.empty(sources.shape[0], dtype=np.int32)
        np.ndarray[COMPLEX64, ndim=2] V = np.empty((N, time), dtype=np.complex128)
        np.ndarray[COMPLEX64, ndim=1] Slp = np.empty(loads.shape[0], dtype=np.complex128)
        np.ndarray[COMPLEX64, ndim=1] Z = np.empty(n.shape[0], dtype=np.complex128)
        np.ndarray[INT32, ndim=1] nl = np.empty(loads.shape[0], dtype=np.int32)
        
        np.ndarray[INT32, ndim=1] n1 = np.empty(n.shape[0], dtype=np.int32)
        np.ndarray[INT32, ndim=1] n2 = np.empty(n.shape[0], dtype=np.int32)

        np.ndarray[FLOAT64, ndim=1] loss = np.zeros(time, dtype=np.float64)
        np.ndarray[FLOAT64, ndim=1] pgen = np.zeros(time, dtype=np.float64)
        np.ndarray[FLOAT64, ndim=1] smin = np.zeros(time, dtype=np.float64)
        np.ndarray[INT32, ndim=1] idx
        np.ndarray[COMPLEX64, ndim=1] Ss = np.empty(N, dtype=np.complex128)
        np.ndarray[INT32, ndim=1] nc = None  # Declarado pero no asignado
        np.ndarray[COMPLEX64, ndim=1] Yc = None

        np.ndarray[COMPLEX64, ndim=1] Sl, I, U
        COMPLEX64 S2, Sg, SSg, dS, V1, V2, ZSconj, Ival, dcomp

    # Asignación de valores a los arrays
    Sl = np.empty(Slp.shape[0], dtype=np.complex128)
    U = np.empty(N, dtype=np.complex128)
    I = np.empty(N, dtype=np.complex128)
    idx = np.empty(loads.shape[0], dtype=np.int32)

    with nogil:
        for i in prange(n.shape[0]):
            n1[i] = <INT32>n[i, 0] - 1
            n2[i] = <INT32>n[i, 1] - 1

        for i in prange(sources.shape[0]):
            s[i] = <INT32>sources[i, 0] - 1

        for i in range(loads.shape[0]):
            Slp[i] = <COMPLEX64>loads[i, 1] + 1j * <COMPLEX64>loads[i, 2]

        for i in range(n.shape[0]):
            Z[i] = <COMPLEX64>n[i, 2] + 1j * <COMPLEX64>n[i, 3]

        for i in range(loads.shape[0]):
            nl[i] = <INT32>loads[i, 0] - 1
    
    for i in range(N):
            for j in range(time):
                V[i, j] = 1.0 + 0.0j

    if caps_present:
        nc = np.asarray(caps[:, 0] - 1, dtype=np.int32)
        Yc = 1j * np.asarray(caps[:, 1], dtype=np.float64)

    for t in range(time):
        for i in range(s.shape[0]):
            V[s[i], t] = <COMPLEX64>sources[i, 2]
        for i in range(Slp.shape[0]):
            Sl[i] = Slp[i]

        if time > 1:
            
            for i in range(loads.shape[0]):
                idx[i] = <INT32>loads[i, 3] - 1

            for i in range(Slp.shape[0]):
                Sl[i] *= graphs[t, idx[i]]
        eps = 1.0
        it = 0
        while eps > lftol and it < lfmaxit:
            it += 1
            for i in range(N):
                U[i] = V[i, t]


            if lfmethod == "radial2":
                for i in range(N):
                    I[i] = 0.0 + 0.0j
                for i in range(nl.shape[0]):
                    I[nl[i]] += (Sl[i] / V[nl[i], t]).conjugate()

                if caps_present:
                    I[nc] += V[nc, t] * Yc

                for i in range(M1 - 1, -1, -1):
                    I[n1[i]] += I[n2[i]]

                smin[t] = 1e9
                for i in range(M1):
                    
                    V[n2[i], t] = V[n1[i], t] - Z[i] * I[n2[i]]
                    S2 = V[n2[i], t] * I[n2[i]].conjugate()
                    SI = (V[n1[i], t].real**2 + V[n1[i], t].imag**2) - 4 * (Z[i] * S2.conjugate()).real
                    if SI < smin[t]:
                        smin[t] = SI

                Sg = 0.0 + 0.0j
                for j in range(s.shape[0]):
                    Sg += V[s[j], t] * I[s[j]].conjugate()


            elif lfmethod == "radial1":
                for i in range(N):
                    Ss[i] = 0.0 + 0.0j
                
                for i in range(nl.shape[0]):
                    Ss[nl[i]] = Sl[i]
                if caps_present:
                    Ss[nc] += abs(V[nc, t]) ** 2 * (Yc).conjugate
                
                for i in range(M1 - 1, -1, -1):
                    Ival = (Ss[n2[i]] / V[n2[i], t]).conjugate()
                    dcomp = Z[i] * (Ival.real**2 + Ival.imag**2)
                    Ss[n1[i]] += Ss[n2[i]] + dcomp

                for i in range(M1):
                    V1 = V[n1[i], t]
                    ZSconj = -Z[i] * (Ss[n2[i]]).conjugate()
                    V1abs = abs(V1)
                    aux = (V1abs / 2) ** 2 + ZSconj.real
                    if aux < 0:
                        return False, 2, None, None, None, None
                    V2 = V1abs / 2 + aux**0.5 + 1j * ZSconj.imag / V1abs
                    V[n2[i], t] = V2 * V1 / V1abs

                Sg = 0.0 + 0.0j
                for j in range(s.shape[0]):
                    Sg += Ss[s[j]]

            else:
                return False, 3, None, None, None, None

            eps = 0.0
            for i in range(N):
                tmp = abs(V[i, t] - U[i])
                if tmp > eps:
                    eps = tmp


        if eps > lftol:
            return False, 1, None, None, None, None

        SSg = Sg
        dS = SSg
        for i in range(Sl.shape[0]):
            dS -= Sl[i]

        loss[t] = dS.real
        pgen[t] = SSg.real

    return True, 0, V, loss, pgen, smin
