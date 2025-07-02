# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
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
        int M1 = n.shape[0]
        int t, i, it, n1, n2
        double eps, SI, V1abs, aux
        bint caps_present = caps is not None

        # Arrays construidos correctamente con operaciones NumPy
        np.ndarray[INT32, ndim=1] s = np.asarray(sources[:, 0] - 1, dtype=np.int32)
        np.ndarray[COMPLEX64, ndim=2] V = np.ones((N, time), dtype=np.complex128)
        np.ndarray[COMPLEX64, ndim=1] Slp = np.asarray(loads[:, 1], dtype=np.complex128) + 1j * np.asarray(loads[:, 2], dtype=np.complex128)
        np.ndarray[COMPLEX64, ndim=1] Z = np.asarray(n[:, 2], dtype=np.complex128) + 1j * np.asarray(n[:, 3], dtype=np.complex128)
        np.ndarray[INT32, ndim=1] nl = np.asarray(loads[:, 0] - 1, dtype=np.int32)

        np.ndarray[FLOAT64, ndim=1] loss = np.zeros(time, dtype=np.float64)
        np.ndarray[FLOAT64, ndim=1] pgen = np.zeros(time, dtype=np.float64)
        np.ndarray[FLOAT64, ndim=1] smin = np.zeros(time, dtype=np.float64)

        np.ndarray[INT32, ndim=1] nc = None
        np.ndarray[COMPLEX64, ndim=1] Yc = None

        np.ndarray[COMPLEX64, ndim=1] Sl, I, Ss, U
        COMPLEX64 S2, Sg, SSg, dS, V1, V2, ZSconj, Ival, dcomp

    if caps_present:
        nc = np.asarray(caps[:, 0] - 1, dtype=np.int32)
        Yc = 1j * np.asarray(caps[:, 1], dtype=np.float64)

    for t in range(time):
        V[s, t] = sources[:, 2]
        Sl = Slp.copy()

        if time > 1:
            idx = np.asarray(loads[:, 3] - 1, dtype=np.int32)
            Sl *= graphs[t, idx]

        eps = 1.0
        it = 0
        while eps > lftol and it < lfmaxit:
            it += 1
            U = V[:, t].copy()

            if lfmethod == "radial2":
                I = np.zeros(N, dtype=np.complex128)
                I[nl] += np.conj(Sl / V[nl, t])
                if caps_present:
                    I[nc] += V[nc, t] * Yc

                for i in range(M1 - 1, -1, -1):
                    n1 = int(n[i, 0]) - 1
                    n2 = int(n[i, 1]) - 1
                    I[n1] += I[n2]

                smin[t] = 1e9
                for i in range(M1):
                    n1 = int(n[i, 0]) - 1
                    n2 = int(n[i, 1]) - 1
                    V[n2, t] = V[n1, t] - Z[i] * I[n2]
                    S2 = V[n2, t] * np.conj(I[n2])
                    SI = (V[n1, t].real**2 + V[n1, t].imag**2) - 4 * (Z[i] * S2.conjugate()).real
                    if SI < smin[t]:
                        smin[t] = SI

                Sg = np.sum(V[s, t] * np.conj(I[s]))

            elif lfmethod == "radial1":
                Ss = np.zeros(N, dtype=np.complex128)
                Ss[nl] = Sl
                if caps_present:
                    Ss[nc] += np.abs(V[nc, t]) ** 2 * np.conj(Yc)

                for i in range(M1 - 1, -1, -1):
                    n1 = int(n[i, 0]) - 1
                    n2 = int(n[i, 1]) - 1
                    Ival = np.conj(Ss[n2] / V[n2, t])
                    dcomp = Z[i] * (Ival.real**2 + Ival.imag**2)
                    Ss[n1] += Ss[n2] + dcomp

                for i in range(M1):
                    n1 = int(n[i, 0]) - 1
                    n2 = int(n[i, 1]) - 1
                    V1 = V[n1, t]
                    ZSconj = -Z[i] * np.conj(Ss[n2])
                    V1abs = np.abs(V1)
                    aux = (V1abs / 2) ** 2 + ZSconj.real
                    if aux < 0:
                        return False, 2, None, None, None, None
                    V2 = V1abs / 2 + aux**0.5 + 1j * ZSconj.imag / V1abs
                    V[n2, t] = V2 * V1 / V1abs

                Sg = np.sum(Ss[s])
            else:
                return False, 3, None, None, None, None

            eps = np.max(np.abs(V[:, t] - U))

        if eps > lftol:
            return False, 1, None, None, None, None

        SSg = Sg
        dS = SSg - np.sum(Sl)
        loss[t] = dS.real
        pgen[t] = SSg.real

    return True, 0, V, loss, pgen, smin
