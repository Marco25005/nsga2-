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
                int k) -> (bint, int):

    cdef int m = 0
    cdef int con_idx
    cdef np.ndarray[np.int32_t, ndim=1] con1, con2, con
    cdef np.ndarray[np.float64_t, ndim=1] tmp

    # Comparación en columnas 0 y 1 casteando a int
    con1 = np.where((flag == 1) & (n[:, 0].astype(np.int32) == sk))[0].astype(np.int32)
    con2 = np.where((flag == 1) & (n[:, 1].astype(np.int32) == sk))[0].astype(np.int32)
    con = np.concatenate((con1, con2)).astype(np.int32)

    if con.shape[0] == 0:
        return True, m

    if con2.shape[0] > 0:
        tmp = n[con2, 0].copy()
        n[con2, 0] = n[con2, 1]
        n[con2, 1] = tmp

    flag[con] = 0

    if np.any(circ[n[con, 1].astype(np.int32), 0] != 0):
        return False, m

    circ[n[con, 1].astype(np.int32), 0] = k

    for con_idx in con:
        m += 1
        order[con_idx] = m
        ok, m_sub = findcircuit(n, circ, order, flag,
                                <int>n[con_idx, 1], k)
        if not ok:
            return False, m + m_sub
        m += m_sub

    return True, m
