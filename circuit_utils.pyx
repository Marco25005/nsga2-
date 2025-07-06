# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython

@cython.cdivision(True)
def findcircuit(np.ndarray[np.float64_t, ndim=2] n,
                np.ndarray[np.int32_t, ndim=2] circ,
                np.ndarray[np.int32_t, ndim=1] order,
                np.ndarray[np.uint8_t, ndim=1] flag,
                int sk,
                int k,
                list m_list) -> bint: # m_list contendrá [m_value]

    cdef int con_idx
    cdef np.ndarray[np.int32_t, ndim=1] con1, con2, con
    cdef np.ndarray[np.float64_t, ndim=1] tmp

    con1 = np.where((flag == 1) & (n[:, 0].astype(np.int32) == sk))[0].astype(np.int32)
    con2 = np.where((flag == 1) & (n[:, 1].astype(np.int32) == sk))[0].astype(np.int32)
    con = np.concatenate((con1, con2)).astype(np.int32)

    if con.shape[0] == 0:
        return True

    if con2.shape[0] > 0:
        tmp = n[con2, 0].copy()
        n[con2, 0] = n[con2, 1]
        n[con2, 1] = tmp

    flag[con] = 0

    # Asegúrate de que circ se indexe correctamente, parece que circ[n[con, 1].astype(np.int32), 0]
    # implica que circ es una matriz 2D. En MATLAB 'circ(index)' es un vector.
    # Si circ en Python es un vector, deberías usar circ[n[con, 1].astype(np.int32)].
    # Si es una matriz 2D y la segunda dimensión es 1 (e.g., circ.shape[1] == 1), está bien.
    if np.any(circ[n[con, 1].astype(np.int32), 0] != 0):
        return False

    circ[n[con, 1].astype(np.int32), 0] = k

    for con_idx in con:
        m_list[0] += 1 # Incrementa el valor en la lista
        order[con_idx] = m_list[0] # Usa el valor actualizado
        
        ok = findcircuit(n, circ, order, flag,
                         <int>n[con_idx, 1], k, m_list) # Pasa la lista mutable
        if not ok:
            return False

    return True