from numpy.polynomial import Polynomial as poly
from control_theory import *
from random import randint

def main():
    open_loop_poles = poles_generator()
    open_loop_nominator = poly([randint(1, 10)])  # open loop nominator === closed loop nominator
    open_loop_denominator = poly.fromroots(open_loop_poles) # poly([0, 5, 6, 1])
    open_loop_zeros = open_loop_nominator.roots()
    closed_loop_denominator = open_loop_denominator + open_loop_nominator
    closed_loop_poles = closed_loop_denominator.roots()
    print_uncompensated_tf(open_loop_nominator, open_loop_poles, open_loop_zeros, closed_loop_denominator, closed_loop_poles)
    
    # check stability
    stable = check_stability(closed_loop_poles)
    print("stability: " + str(stable))
    if not stable:
        return

    real_closed_loop_poles, imaginary_closed_loop_poles, complex_closed_loop_poles = sort_roots(closed_loop_poles)
    print("complex closed loop poles: " + str(complex_closed_loop_poles))
    print("real closed loop poles: " + str(real_closed_loop_poles))
    print("imaginary closed loop poles: " + str(imaginary_closed_loop_poles)+"\n")

    # check order
    if len(closed_loop_poles) == 2:
        closed_loop_poles_significant = closed_loop_poles
    elif len(closed_loop_poles) == 3:
        # check negligibility
        negligible, closed_loop_poles_significant = check_negligibility(closed_loop_poles)
        print("negligibility: " + str(negligible)+"\n")
        print("significant closed loop poles: " + str(closed_loop_poles_significant)+"\n")

    # 2nd order system (or approximation)
    if (len(closed_loop_poles)==2) or negligible:
        type = check_poles_type(closed_loop_poles_significant)
        try:    
            c, PO, ts = parameters_switch(type, closed_loop_poles_significant)
        except:
            return
        print("closed loop poles: " + type + " => " + str(closed_loop_poles_significant)+"\n")

    # specifications
    PO_max, ts_max = specifications_generator(PO, ts)
    c_min, phase_deficiency_max, sigma_d_min = specifications(PO_max, ts_max)
    print_specifications(PO_max, ts_max, c_min, phase_deficiency_max, sigma_d_min)

    # phase-lead componsator
    are_specifiactions_met = False
    iteration = 1
    while not are_specifiactions_met:
        print("\n*************** TRIAL NUMBER: "+ str(iteration) +" ***************")
        pd, pd_phase = pd_complex_pair_generator(c_min, phase_deficiency_max)
        print("pd: " + str(pd))
        print("pd phase: "+str(pd_phase))
        try:
            new_c, new_PO, new_ts = parameters_switch(type, [pd])
        except:
            return
        is_PO_accepted = "accepted" if ((new_PO<=PO_max)and(new_PO>=0)) else "not accepted"
        is_ts_accepted = "accepted" if ((new_ts <= ts_max)and(new_ts>=0))else "not accepted"
        print("new c: " + str(new_c))
        print("new PO: " + str(new_PO) + " % => " + is_PO_accepted)
        print("new ts: " + str(new_ts) + " s => " + is_ts_accepted)
        
        if is_PO_accepted=="accepted" and is_ts_accepted=="accepted":
            are_specifiactions_met = True
        iteration += 1
        if iteration==1000:  # trial limit for one set of data
            print("\nProgram ran for 1000 times without meeting requirements\n")
            return
        print("***********************************************") 

    required_additional_phase = required_phase(pd, open_loop_zeros, open_loop_poles)
    print("\nrequired additional phase: " + str(required_additional_phase) + " °")

    # choosing zc such as an OL pole is cancelled 
    zc, pc, kc = compensator_zpk(type, pd, open_loop_poles, required_additional_phase, open_loop_nominator, open_loop_denominator)
    print("Gc(s) = " + str(kc) + " * (s+" + str(-zc) + ") / (s+" + str(-pc) +")")

    if pc > 10 or pc < -10 or kc > 10 or kc < -10:
        return

    tf_nominator, tf_denominator = closed_loop_tf(type, zc, kc, pc, open_loop_nominator, open_loop_poles)
    print("")
    print("closed loop TF:")
    print("nominator: " + str(tf_nominator))
    print("denominator: " + str(tf_denominator))
    print("closed-loop zeros: " + str(tf_nominator.roots()))
    print("closed-loop poles: " + str(tf_denominator.roots()))
    
    tf_poles = tf_denominator.roots()
    line = [open_loop_nominator.coef[0], open_loop_denominator.coef[0], open_loop_denominator.coef[1], open_loop_denominator.coef[2], open_loop_denominator.coef[3], closed_loop_denominator.coef[0], closed_loop_denominator.coef[1], closed_loop_denominator.coef[2], closed_loop_denominator.coef[3], closed_loop_poles[0].real, closed_loop_poles[0].imag, closed_loop_poles[1].real, closed_loop_poles[1].imag, closed_loop_poles[2].real, closed_loop_poles[2].imag, PO_max, ts_max, c_min, phase_deficiency_max, sigma_d_min, pd.real, pd.imag, pd_phase, new_c, new_PO, new_ts, required_additional_phase, zc, pc, kc, tf_nominator.coef[0], tf_denominator.coef[0], tf_denominator.coef[1], tf_denominator.coef[2], tf_denominator.coef[3], tf_poles[0].real, tf_poles[0].imag, tf_poles[1].real, tf_poles[1].imag, tf_poles[2].real, tf_poles[2].imag]
    write_results(line, "data/data.csv")
    print("")
    
    

if __name__ == "__main__":
    for i in range(100000):
        main()
    