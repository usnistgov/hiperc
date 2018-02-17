#!/usr/bin/python
# -*- coding: utf-8 -*-

## Sympy code to generate expressions for PFHub Problem 7 (MMS)

from sympy import Symbol, symbols, simplify
from sympy import Eq, sin, cos, cosh, sinh, tanh, sqrt
from sympy.physics.vector import divergence, gradient, dynamicsymbols, ReferenceFrame, time_derivative
from sympy.printing import ccode, pprint
from sympy.abc import kappa, S, t
import string

# Spatial coordinates: x=R[0], y=R[1], z=R[2]
R = ReferenceFrame('R')

# sinusoid amplitudes
A1, A2 = symbols('A1 A2')
B1, B2 = symbols('B1 B2')
C2 = symbols('C2')

# Define interface offset (alpha)
alpha = 0.25 + A1 * t * sin(B1 * R[0]) \
             + A2     * sin(B2 * R[0] + C2 * t)
print("\nInterface displacement:")
pprint(Eq(symbols('alpha'), alpha))

# Define the solution equation (eta)
eta = 0.5 * (1 - tanh((R[1] - alpha) / sqrt(2*kappa)))
print("\nManufactured Solution:")
pprint(Eq(symbols('eta'), eta))

# Compute the initial condition
print("\nInitial condition:")
pprint(Eq(symbols('eta0'), eta.subs(t, 0)))

# Compute the source term from the equation of motion
S = simplify(time_derivative(eta, R) + 4 * eta * (eta - 1) * (eta - 1/2) - kappa * divergence(gradient(eta, R), R))
print("\nSource term:")
pprint(Eq(symbols('S'), S))


# === Check Results against @stvdwtt ===
dadx = A1 * B1 * t * cos(B1 * R[0]) + A2 * B2 * cos(B2*R[0] + C2*t)
dadt = A1 * sin(B1*R[0])            + A2 * C2 * cos(B2*R[0] + C2*t)
d2adx2 = -A1 * B1**2 * t * sin(B1*R[0]) - A2 * B2**2 * sin(B2*R[0] + C2*t)
sech = 1 / cosh((R[1]-alpha)/sqrt(2*kappa))
Sdw = sech**2 / sqrt(16*kappa) * (-2*sqrt(kappa)*tanh((R[1]-alpha)/sqrt(2*kappa)) \
                                  * (dadx)**2 + sqrt(2)*(dadt - kappa*d2adx2))

print("@tkphd and @stvdwtt agree?")
notZero = Sdw - S
pprint("True" if (not notZero) else notZero)

print("\nC codes (without types):")
CA = ccode(alpha).replace('R_x','x').replace('R_y','y')
CH = ccode(eta).replace('R_x','x').replace('R_y','y')
C0 = ccode(eta.subs(t, 0)).replace('R_x','x').replace('R_y','y')
CS = ccode(S).replace('R_x','x').replace('R_y','y')

print("\nalpha(x, y, t) {")
pprint(CA)
print("}\n\nmanufacturedSolution(x, y, t) {")
pprint(CH)
print("}\n\ninitialCondition(x, y, t) {")
pprint(C0)
print("}\n\nsourceTerm(x, y, t) {")
pprint(CS)
print("}")
