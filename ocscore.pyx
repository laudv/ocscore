#cython: language_level=3

import numpy as np

#cimport numpy as np
cimport cython
from cython.parallel import prange, parallel

from libc.stdlib cimport abort, malloc, free
from libc.stdint cimport int8_t,  int32_t,  int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

cdef extern from "immintrin.h":
    ctypedef struct __m256i:
        pass

    __m256i _mm256_lddqu_si256(__m256i *) nogil
    void _mm256_storeu_si256 (__m256i *, __m256i) nogil
    __m256i _mm256_setzero_si256() nogil
    __m256i _mm256_xor_si256(__m256i, __m256i) nogil
    __m256i _mm256_adds_epu8(__m256i, __m256i) nogil
    __m256i _mm256_min_epu8(__m256i, __m256i) nogil
    __m256i _mm256_set1_epi8(int8_t) nogil
    __m256i _mm256_cmpeq_epi8(__m256i, __m256i) nogil
    __m256i _mm256_andnot_si256(__m256i, __m256i) nogil
    int32_t _mm256_extract_epi8(__m256i, int32_t) nogil

#------------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mapids_uint8_(int32_t[:,:] leafs, uint8_t[:,:] leaf2id, uint8_t[:,:] out) nogil:
    for j in xrange(leafs.shape[1]):
        for i in xrange(leafs.shape[0]):
            out[i, j] = leaf2id[j, leafs[i, j]]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mapids_uint16_(int32_t[:,:] leafs, uint16_t[:,:] leaf2id, uint16_t[:,:] out) nogil:
    for j in xrange(leafs.shape[1]):
        for i in xrange(leafs.shape[0]):
            out[i, j] = leaf2id[j, leafs[i, j]]

# leaf2id[tree_id, node_id] = leaf ID
def mapids(leafs, leaf2id, dtype):
    ids = np.zeros(leafs.shape, order="F", dtype=dtype)
    if dtype == np.uint8:
        mapids_uint8_(leafs, leaf2id, ids)
    elif dtype == np.uint16:
        mapids_uint16_(leafs, leaf2id, ids)
    else:
        raise ValueError("invalid dtype")
    return ids


#- UINT8 ----------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint32_t _ocscore_colmaj8(uint8_t[:, :] refset, uint8_t[:] test) nogil:
    cdef int i
    cdef int j
    cdef __m256i vcnt = _mm256_set1_epi8(-1) # 255
    cdef __m256i s # sum
    cdef __m256i x
    cdef uint32_t min_cnt = ~0
    for i in xrange(0, (refset.shape[0]//32)*32, 32): # avx2 256-bit registers
        s = _mm256_setzero_si256()
        for j in xrange(refset.shape[1]):
            x = _mm256_lddqu_si256(<__m256i *>&refset[i, j])
            x = _mm256_xor_si256(x, _mm256_set1_epi8(test[j]))
            x = _mm256_cmpeq_epi8(x, _mm256_setzero_si256())
            # NOT(x) & 0b00000001 ==> 0b1 if allzero, else 0
            x = _mm256_andnot_si256(x, _mm256_set1_epi8(1))
            s = _mm256_adds_epu8(s, x)
        vcnt = _mm256_min_epu8(vcnt, s)

    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 0))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 1))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 2))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 3))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 4))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 5))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 6))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 7))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 8))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 9))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 10))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 11))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 12))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 13))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 14))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 15))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 16))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 17))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 18))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 19))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 20))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 21))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 22))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 23))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 24))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 25))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 26))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 27))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 28))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 29))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 30))
    min_cnt = min(min_cnt, _mm256_extract_epi8(vcnt, 31))

    cdef uint8_t tmp
    for i in xrange((refset.shape[0]//32)*32, refset.shape[0]):
        tmp = 0
        for j in xrange(refset.shape[1]):
            tmp += (refset[i, j] ^ test[j]) > 0
        min_cnt = min(min_cnt, tmp)

    return min_cnt

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _ocscores_colmaj8(uint8_t[:, :] refset, uint8_t[:, :] test, uint32_t[:] out) nogil:
    cdef int i
    for i in prange(test.shape[0], nogil=True):
        out[i] = _ocscore_colmaj8(refset, test[i,:])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _ocscore_colmaj_topk(uint8_t[:, :] refset, uint8_t[:] test, uint8_t *buf) nogil:
    cdef int i
    cdef int j
    cdef __m256i s # sum
    cdef __m256i x
    for i in xrange(0, (refset.shape[0]//32)*32, 32): # avx2 256-bit registers
        s = _mm256_setzero_si256()
        for j in xrange(refset.shape[1]):
            x = _mm256_lddqu_si256(<__m256i *>&refset[i, j])
            x = _mm256_xor_si256(x, _mm256_set1_epi8(test[j]))
            x = _mm256_cmpeq_epi8(x, _mm256_setzero_si256())
            # NOT(x) & 0b00000001 ==> 0b1 if allzero, else 0
            x = _mm256_andnot_si256(x, _mm256_set1_epi8(1))
            s = _mm256_adds_epu8(s, x)
        _mm256_storeu_si256(<__m256i *>&buf[i], s)

    cdef uint8_t tmp
    for i in xrange((refset.shape[0]//32)*32, refset.shape[0]):
        tmp = 0
        for j in xrange(refset.shape[1]):
            tmp += (refset[i, j] ^ test[j]) > 0
        buf[i] = tmp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _ocscores_colmaj_topk(uint8_t[:, :] refset, uint8_t[:, :] test,
        uint32_t[:, :] indexes, uint8_t[:, :] scores):
    cdef int i
    cdef int k
    cdef int j
    cdef uint8_t * local_buf
    cdef int top_k
    cdef uint8_t m
    cdef int mj

    top_k = <int> indexes.shape[1]

    with nogil, parallel():
        local_buf = <uint8_t *> malloc(sizeof(uint8_t) * refset.shape[0])
        if local_buf is NULL:
            abort()

        # share the work using the thread-local buffer(s)
        for i in prange(test.shape[0], schedule="guided"):
            _ocscore_colmaj_topk(refset, test[i,:], local_buf)
            # sort for top_k best elements only
            for k in xrange(top_k):
                m = 255
                mj = 0
                for j in xrange(refset.shape[0]):
                    if m > local_buf[j]:
                        m = local_buf[j]
                        mj = j
                indexes[i, k] = mj
                scores[i, k] = m
                local_buf[mj] = 255

        free(local_buf)

#- UINT16 --------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint32_t _ocscore_colmaj16(uint16_t[:, :] refset, uint16_t[:] test) nogil:
    cdef int i
    cdef int j
    cdef uint32_t min_cnt = ~0

    cdef uint16_t tmp
    for i in xrange(refset.shape[0]):
        tmp = 0
        for j in xrange(refset.shape[1]):
            tmp += (refset[i, j] ^ test[j]) > 0
        min_cnt = min(min_cnt, tmp)

    return min_cnt

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _ocscores_colmaj16(uint16_t[:, :] refset, uint16_t[:, :] test, uint32_t[:] out) nogil:
    cdef int i
    for i in prange(test.shape[0], nogil=True):
        out[i] = _ocscore_colmaj16(refset, test[i,:])




#- ENTRY POINTS ---------------------------------------------------------------

def _validate_input(refset, test):
    if not (
            (refset.dtype == np.uint8  and test.dtype == np.uint8)
        or  (refset.dtype == np.uint16 and test.dtype == np.uint16)):
        raise ValueError("must be byte or short matrices") 
    if refset.shape[1] != test.shape[1]:
        raise ValueError("shapes do not match")
    if refset.shape[1] >= 255:
        raise ValueError("possible overflow due to too many trees and use of uint8")

def _prep_input(refset, test, order, dtype):
    sh = refset.shape
    sz = dtype.itemsize
    if sh[1] % sz != 0:
        z = np.zeros((sh[0], (sh[1]//sz+1)*sz-sh[1]), dtype)
        refset = np.concatenate((refset, z), axis=1)
        z = np.zeros((test.shape[0], (sh[1]//sz+1)*sz-sh[1]), dtype)
        test = np.concatenate((test, z), axis=1)
    refset = np.array(refset.view(dtype), order=order)
    test = np.array(test.view(dtype), order=order)
    return refset, test

def ocscores(refset, test):
    _validate_input(refset, test)
    dtype = refset.dtype
    refset, test = _prep_input(refset, test, order="F", dtype=dtype)

    out = np.zeros(test.shape[0], dtype=np.uint32)

    if dtype == np.uint8:
        _ocscores_colmaj8(refset, test, out)
    elif dtype == np.uint16:
        _ocscores_colmaj16(refset, test, out)
    else:
        raise ValueError("invalid dtype of refset")

    return out

def ocscores_topk(refset, test, k):
    _validate_input(refset, test)
    dtype = refset.dtype
    refset, test = _prep_input(refset, test, order="F", dtype=dtype)

    indexes = np.zeros((test.shape[0], k), dtype=np.uint32)
    scores = np.zeros((test.shape[0], k), dtype=np.uint8)

    if dtype == np.uint8:
        _ocscores_colmaj_topk(refset, test, indexes, scores)
    else:
        raise ValueError("invalid dtype of refset (no uint16 implementation yet)")

    return indexes, scores
