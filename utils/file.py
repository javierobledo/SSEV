from scipy import sparse
import tables as tb
from numpy import array


def load_sparse_mat(name, store='store.h5'):
    with tb.open_file(store) as f:
        pars = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            pars.append(getattr(f.root, '%s_%s' % (name, par)).read())
    m = sparse.csr_matrix(tuple(pars[:3]), shape=pars[3])
    return m


def store_sparse_mat(m, name, store='store.h5'):
    msg = "This code only works for csr matrices"
    assert (m.__class__ == sparse.csr.csr_matrix), msg
    with tb.open_file(store, 'a') as f:
        for par in ('data', 'indices', 'indptr', 'shape'):
            full_name = '%s_%s' % (name, par)
            try:
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass

            arr = array(getattr(m, par))
            atom = tb.Atom.from_dtype(arr.dtype)
            ds = f.create_carray(f.root, full_name, atom, arr.shape)
            ds[:] = arr