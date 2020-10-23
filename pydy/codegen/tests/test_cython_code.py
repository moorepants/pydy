#!/usr/bin/env python

import os

import numpy as np
import sympy as sm

from ...models import multi_mass_spring_damper, n_link_pendulum_on_cart
from ..cython_code import CythonMatrixGenerator


class TestCythonMatrixGenerator(object):

    def setup(self):

        self.prefix = 'boogly_bee'

        sys = multi_mass_spring_damper(6, True, True)

        self.matrices = (sys.eom_method.mass_matrix,
                         sys.eom_method.forcing)

        self.arguments = (sys.constants_symbols,
                          sys.coordinates,
                          sys.speeds,
                          sys.specifieds_symbols)

        self.generator = CythonMatrixGenerator(self.arguments,
                                               self.matrices, self.prefix)

    def test_generate_code_blocks(self):

        expected = {}

        expected['header_args'] = \
"""\
                  double* input_0,
                  double* input_1,
                  double* input_2,
                  double* input_3,
                  double* output_0,
                  double* output_1\
"""

        expected['python_args'] = \
"""\
         np.ndarray[np.double_t, ndim=1, mode='c'] input_0,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_1,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_2,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_3,
         np.ndarray[np.double_t, ndim=1, mode='c'] output_0,
         np.ndarray[np.double_t, ndim=1, mode='c'] output_1\
"""

        expected['c_args'] = \
"""\
             <double*> input_0.data,
             <double*> input_1.data,
             <double*> input_2.data,
             <double*> input_3.data,
             <double*> output_0.data,
             <double*> output_1.data\
"""

        expected['output'] = \
"""\
            output_0.reshape(6, 6),
            output_1\
"""

        self.generator._generate_code_blocks()

        for k, v in self.generator.code_blocks.items():
            assert v == expected[k]

    def test_doprint(self):

        expected_pyx_source = \
"""\
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "boogly_bee_c.h":
    void evaluate(
                  double* input_0,
                  double* input_1,
                  double* input_2,
                  double* input_3,
                  double* output_0,
                  double* output_1
                 )

@cython.boundscheck(False)
@cython.wraparound(False)
def eval(
         np.ndarray[np.double_t, ndim=1, mode='c'] input_0,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_1,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_2,
         np.ndarray[np.double_t, ndim=1, mode='c'] input_3,
         np.ndarray[np.double_t, ndim=1, mode='c'] output_0,
         np.ndarray[np.double_t, ndim=1, mode='c'] output_1
        ):

    evaluate(
             <double*> input_0.data,
             <double*> input_1.data,
             <double*> input_2.data,
             <double*> input_3.data,
             <double*> output_0.data,
             <double*> output_1.data
            )

    return (
            output_0.reshape(6, 6),
            output_1
           )\
"""

        expected_setup_py_source = """\
#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize
import numpy

extension = Extension(name="boogly_bee",
                      sources=["boogly_bee.pyx",
                               "boogly_bee_c.c"],
                      include_dirs=[numpy.get_include()])

setup(name="boogly_bee",
      ext_modules=cythonize([extension]))\
"""

        setup, pyx, c_header, c_source = self.generator.doprint()

        assert setup == expected_setup_py_source
        assert pyx == expected_pyx_source

    def test_write(self):

        setup, pyx, c_header, c_source = self.generator.doprint()

        self.generator.write()

        with open(self.prefix + '_c.h') as f:
            assert f.read() == c_header

        with open(self.prefix + '_c.c') as f:
            assert f.read() == c_source

        with open(self.prefix + '_setup.py') as f:
            assert f.read() == setup

        with open(self.prefix + '.pyx') as f:
            assert f.read() == pyx

    def test_compile(self):

        f = self.generator.compile()

        subs = {}

        args = []
        for argset in self.arguments:
            vals = np.random.random(len(argset))
            args.append(vals)
            for arg, val in zip(argset, vals):
                subs[arg] = val

        for matrix in self.matrices:
            nr, nc = matrix.shape
            args.append(np.empty(nr * nc, dtype=float))

        for output, expected in zip(f(*args), self.matrices):
            try:
                expected = sm.matrix2numpy(expected.subs(subs),
                                           dtype=float).squeeze()
            except TypeError:
                # dtype kwarg in not supported in earlier SymPy versions
                expected = np.asarray(sm.matrix2numpy(expected.subs(subs)),
                                      dtype=float).squeeze()

            np.testing.assert_allclose(output, expected)

    def teardown(self):

        for suffix in ['_c.h', '_c.c', '_setup.py', '.pyx']:
            filename = self.prefix + suffix
            if os.path.isfile(filename):
                os.remove(filename)


def test_cse_equivalency():

    sys = n_link_pendulum_on_cart(n=10, cart_force=True, joint_torques=True)

    M = sys.eom_method.mass_matrix_full
    F = sys.eom_method.forcing_full

    x = sys.states
    r = sys.specifieds_symbols
    p = sys.constants_symbols

    x_vals = np.random.random(len(x))
    r_vals = np.random.random(len(r))
    p_vals = np.random.random(len(p))

    subs_dict = dict(zip(x, x_vals))
    subs_dict.update(dict(zip(r, r_vals)))
    subs_dict.update(dict(zip(p, p_vals)))

    gen1 = CythonMatrixGenerator([x, r, p], [M, F], cse=False)
    c_eval_M_F_no_cse = gen1.compile()
    M1 = np.zeros(len(x)*len(x))
    F1 = np.zeros(len(x))
    c_eval_M_F_no_cse(x_vals, r_vals, p_vals, M1, F1)

    gen2 = CythonMatrixGenerator([x, r, p], [M, F], cse=True)
    c_eval_M_F_with_cse = gen2.compile()
    M2 = np.zeros(len(x)*len(x))
    F2 = np.zeros(len(x))
    c_eval_M_F_with_cse(x_vals, r_vals, p_vals, M2, F2)

    # check xreplace for the heck of it too
    M4 = sm.matrix2numpy(M.xreplace(subs_dict), dtype=float)
    F4 = sm.matrix2numpy(F.xreplace(subs_dict), dtype=float)

    # Compare the other forms to the evalf form.
    M3 = sm.matrix2numpy(M.evalf(subs=subs_dict), dtype=float)
    F3 = sm.matrix2numpy(F.evalf(subs=subs_dict), dtype=float)

    np.testing.assert_allclose(M1.reshape((len(x), len(x))), M3)
    np.testing.assert_allclose(F1, np.squeeze(F3))

    np.testing.assert_allclose(M2.reshape((len(x), len(x))), M3)
    np.testing.assert_allclose(F2, np.squeeze(F3))

    np.testing.assert_allclose(M4, M3)
    np.testing.assert_allclose(F4, F3)
