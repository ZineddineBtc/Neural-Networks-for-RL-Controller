import numpy as np
from numpy.polynomial import Polynomial as poly
from numpy.random import rand
from random import uniform
from math import sqrt, exp, pi, acos, tan
import csv

from indexes import indexes

def poles_generator():  # 2 dominant conjugates + negligible pole
    x = -1 * rand(1)
    y = -1 * rand(1)
    z = x - 6  # negligible
    return [x[0], y[0], z[0]]

def sort_roots(roots):
    real_roots = []
    imaginary_roots = []
    complex_roots = [] 
    for root in roots:
        if root.imag == 0:
            real_roots.append(root.real)
        elif root.real == 0:
            imaginary_roots.append(root.imag)
        else:
            complex_roots.append(root)
    return real_roots, imaginary_roots, complex_roots

def poles_pair_to_c_w(real, imag):
    n = real ** 2
    d = n + (imag**2)
    c = sqrt(n/d)
    w = real / (-c)
    return c, w 

def c_to_po(c):
    if c > 1:
        n = pi * (c)
        d = sqrt((c**2)-1)
    else:
        n = pi * (-c)
        d = sqrt(1-(c**2))
    return exp(n/d) * 100

def c_to_ts(w, c):
    return 4 / (c * w)

def check_stability(roots):
    for root in roots:
        if root.real > 0:
            return False
    return True

def check_negligibility(roots):
    if min(roots) <= (max(roots)*5):
        negligible = True
        roots_significant = list(roots)
        roots_significant.remove(min(roots))
    else:
        negligible = False
    return negligible, roots_significant

def check_conjugates(roots):
    complex_conjugate = (roots[0].real != 0) and (roots[0].real == roots[1].real) and (roots[0].imag == -roots[1].imag)
    imaginary_conjugate = (roots[0].real == 0) and (roots[1].real == 0) and (roots[0].imag == -roots[1].imag)
    return complex_conjugate, imaginary_conjugate

def check_repeated(roots):
    return (roots[0].real == roots[1].real) and (roots[0].imag == roots[1].imag)

def check_poles_type(roots_significant):
    hasComplexConjugatePair, hasImaginaryConjugatePair = check_conjugates(roots_significant)
    if hasComplexConjugatePair:
        return "complex"
    elif hasImaginaryConjugatePair:
        return "imaginary"
    elif check_repeated(roots_significant):
        return "repeated"
    else:
        return "distinct"

def parameters_complex_conjugate(roots):
    c, w = poles_pair_to_c_w(roots[0].real, roots[0].imag)
    PO = c_to_po(c)
    ts = c_to_ts(w, c)
    return c, PO, ts

def parameters_imaginary_conjugate(roots):
    c = 0
    w = abs(roots[0])
    ts = 0
    PO = c_to_po(c)
    return c, PO, ts

def parameters_repeated(roots):
    c = 1
    w = -roots[0]
    PO = c_to_po(c)
    ts = c_to_ts(w, c)
    return c, PO, ts 

def parameters_distinct(roots):
    reals = [roots[0].real, roots[1].real]
    a = min(reals)
    b = max(reals)
    k = (2*a)/(a+b)
    c = (2*k)/(k+1)
    w = (a+b)/(-2*c)
    PO = c_to_po(c)
    ts = c_to_ts(w, c)
    return c, PO, ts

def parameters_switch(type, roots_significant):
    if type == "complex":
        c, PO, ts = parameters_complex_conjugate(roots_significant)
    elif type == "imaginary":
        c, PO, ts = parameters_imaginary_conjugate(roots_significant)
    elif type == "repeated":
        c, PO, ts = parameters_repeated(roots_significant)
    elif type == "distinct":
        c, PO, ts = parameters_distinct(roots_significant)
    return c, PO, ts

def specifications_generator(PO, ts):
    PO_max = uniform(1, (PO-0.1))
    if PO_max <= 0:
        PO_max = 0.1
    ts_max = uniform(1, (ts-0.1))
    if ts_max <= 0:
        ts_max = 0.1
    return PO_max, ts_max 

def specifications(PO_max, ts_max):
    c_min = -np.log(PO_max/100)/sqrt(pi*pi + np.log(PO_max/100)**2)
    phase_deficiency_max = acos(c_min)*180/pi  # in degrees
    sigma_d_min = 4 / ts_max
    return c_min, phase_deficiency_max, sigma_d_min

def pd_complex_pair_generator(c_min, phase_deficiency_max):
    real = -1 * uniform(c_min, (c_min+3))
    phase = uniform(0, phase_deficiency_max)
    imaginary = -1 * tan(phase) * real
    pd = real + (imaginary*1j)
    return pd, phase

def required_phase(pd, open_loop_zeros, open_loop_poles):
    angle = 0
    for zero in open_loop_zeros: angle += np.angle((pd-zero), deg=True)
    for pole in open_loop_poles: angle -= np.angle((pd-pole), deg=True)
    return (180 - angle) % 360

def set_zc(type, open_loop_poles):
    if type == "complex":
        for pole in open_loop_poles:
            if pole != max(open_loop_poles) and pole != min(open_loop_poles):
                zc = pole
    elif type == "imaginary":
        zc = -6  # insignificant
    return zc
    
def compensator_zpk(type, pd, open_loop_poles, required_additional_phase, open_loop_nominator, open_loop_denominator):
    zc = set_zc(type, open_loop_poles)
    phase_pc = 90 - required_additional_phase
    pc = pd.real - (pd.imag/tan(phase_pc*pi/180))
    g_OL_of_pd = open_loop_nominator(pd)/open_loop_denominator(pd)
    kc = 1/abs((((pd-zc)/(pd-pc))*g_OL_of_pd))
    return zc, pc, kc

def closed_loop_tf(type, zc, kc, pc, open_loop_nominator, open_loop_poles):
    gc_nominator = poly([kc*(-zc), kc])
    gc_denominator = poly([-zc, 1])
    # tf = (gc*g_OL) / (1 + gc*g_OL)
    tf_nominator = open_loop_nominator * kc
    if type == "complex":
        pole_OL_cancelled_poles = list(open_loop_poles)
        pole_OL_cancelled_poles.remove(zc)
        tf_denominator = [-pc, 1]*poly.fromroots(pole_OL_cancelled_poles) + tf_nominator
    elif type == "imaginary":
        tf_denominator = [-pc, 1]*open_loop_poles + tf_nominator 
    return tf_nominator, tf_denominator

def print_uncompensated_tf(open_loop_nominator, open_loop_poles, open_loop_zeros, closed_loop_denominator, closed_loop_poles):
    print("G(s): (uncompensated TF)")
    print("open loop nominator: " + str(open_loop_nominator) + " (same as closed loop)")
    print("open loop denominator: " + str(open_loop_poles))
    print("open loop zeros: " + str(open_loop_zeros) + " (same as closed loop)")
    print("open loop poles: " + str(open_loop_poles))
    print("closed loop nominator: " + str(open_loop_nominator) + " (same as open loop)")
    print("closed loop denominator: " + str(closed_loop_denominator))
    print("closed loop zeros: " + str(open_loop_zeros) + " (same as open loop)")
    print("closed loop poles: " + str(closed_loop_poles) + "\n")

def print_specifications(PO_max, ts_max, c_min, phase_deficiency_max, sigma_d_min):
    print("**************** SPECIFICATION ****************")
    print("PO max: " + str(PO_max) + " %")
    print("Ts max: " + str(ts_max) + " s")
    print("c min: " + str(c_min))
    print("phase deficiency max: " + str(phase_deficiency_max) + " °")
    print("sigma_d min: " + str(sigma_d_min))
    print("***********************************************")

def write_results(line, csv_name):
    outfile = open(csv_name, "a")
    out = csv.writer(outfile)
    out.writerow(line)  # indexes already written
    outfile.close()

def read_results(path):
    rows = []
    with open(path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        titles = next(csvreader)
        for row in csvreader:
            rows.append(row)
    # rows: 2D list of (row, cols)
    return rows
