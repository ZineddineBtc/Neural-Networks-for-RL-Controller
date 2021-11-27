# Compensator based on Root-Locus

## Initial Descriptions
* System: Y(s) = U(s) * (G(s)/(1+G(s)))
* U(s): input
* Y(s): output
* G(s): uncompensated open-loop transfer function
* G(s)/(1+G(s)): uncompensated closed-loop transfer function

## Parameters
* Damping ratio (c)
* Percentage Overshoot (PO)
* Settling Time (Ts)

## Design General Guidelines
1. Analyze (simulate) the uncompensated system, if it meets the specification , stop here.
2. Tune the compensator parameters (here, by help of the root-locus)
3. Simulate the compensated system, if not satisfactory go back to 2

Note: producing good (excellent) designs require intuition, experience, and practice, there is no general formula, thus it is both an art and science

## Rationale
If intuition and experience are important in designing a compensator, the questen stands: can an artificial neural network learn it?

## Purpose
The Root-Locus--based compensation is about finding a transfer function Gc(s) such that the parameters meet certain wished values.
Let:
* G-cascade(s) = Gc(s)*G(s)
* TF(s) = G-cascade(s)/(1+G-cascade(s))
The compensated system: Y(s) = U(s) * TF(s)

## Procedures
1. Checking stability.
2. Checking system order (3rd order + negligibility => 2nd order).
3. Depending on the type of closed-loop poles of uncompensated closed-loop transfer function, the program calculates the parameters.
4. Parameter specifications (wished parameters) are introduced and relevant values are calculated.
5. The program iterates for a maximum of 1000 loops, each loop generating random poles (The tuning pd) with which it measures the new parameters and check if they meet the specifications. If so, then step 6. If the program loops 1000 times without generating poles meeting the specifications, the script quits and no further calculations are done. 
6. The program calculates the required additional phase for a Phase-Lead compensator to be introduced as Gc(s) and calculates its zpk (zero-pole-gain) depending on the type of the closed-loop poles the uncompensated system.
7. The program calculates the closed-loop transfer function TF(s) such that:
TF(s) = G-cascade(s)/(1+G-cascade(s)) 
TF(s) = Y(s)/U(s)
8. The program writes all of the calculated values in a csv file depending on the type of the closed-loop poles of the uncompensated closed-loop transfer function.

## Artificial Intelligence Development

* The `root_locus_nn.py` script contains the class that permits the building, training, and the use of the a neural network in designing a phase-lead compensator.
* The `experiments.py` script includes the functions encapsulating the experiements done in this work.
* The `main.py` script holds the execution of the finalized model (post experimentation) and visualizes its results.
