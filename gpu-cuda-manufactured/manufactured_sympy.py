#!/usr/bin/python
# -*- coding: utf-8 -*-

# Sympy code to generate expressions for PFHub Problem 7 (MMS)

from sympy import Symbol, symbols, simplify
from sympy import cos, diff, Eq, expand, factor, sin, sqrt, tanh
from sympy.vector import divergence, gradient, CoordSys3D
from sympy.physics.vector import time_derivative, ReferenceFrame
from sympy.printing import ccode, pprint
from sympy.abc import kappa, S, t, X
import string

# Spatial coordinates: x=x[0], y=x[1], etc.
x = CoordSys3D('x')

# sinusoid amplitudes
A1, A2 = symbols('A1 A2')
B1, B2 = symbols('B1 B2')
C2 = symbols('C2')

print("\nInterface displacement:")
alpha = 4**-1 + A1 * t * sin(B1 * x.x) + A2 * sin(B2 * x.x + C2 * t)
pprint(Eq(symbols('alpha'), alpha))

print("\nManufactured Solution:")
Z = (x.y - alpha) / sqrt(2*kappa)
eta = (1 - tanh(Z)) / 2
pprint(Eq(symbols('eta'), eta))

print("\nInitial condition:")
pprint(Eq(symbols('eta0'), eta.subs(t, 0)))

print("\nSource term:")
fprime = diff(X**2 * (X - 1)**2, X).subs(X, eta)
laplacian = divergence(gradient(eta))
dedt = time_derivative(eta, ReferenceFrame('x'))
S = simplify(dedt + fprime - kappa * laplacian)
pprint(Eq(symbols('S'), S))

print("CAS agrees with human source term?")
dadt   =  A1 * sin(B1*x.x)             + A2 * C2 * cos(B2*x.x + C2*t)
dadx   =  A1 * B1 * t * cos(B1 * x.x)  + A2 * B2 * cos(B2*x.x + C2*t)
d2adx2 = -A1 * B1**2 * t * sin(B1*x.x) - A2 * B2**2 * sin(B2*x.x + C2*t)
Sdw = (1 - tanh(Z)**2) / sqrt(16*kappa) * (  sqrt(2)*dadt
                                           - 2*sqrt(kappa) * tanh(Z) * dadx**2
                                           - sqrt(2)*kappa*d2adx2)
notZero = Sdw - S
pprint("True" if (not notZero) else "False")

print("\nC codes (without types):")
print("\nalpha(x, t) {")
pprint(ccode(alpha).replace('x.x','x').replace('x.y','y'))
print("}\n\nmanufacturedSolution(x, y, t) {")
pprint(ccode(eta).replace('x.x','x').replace('x.y','y'))
print("}\n\ninitialCondition(x, y, t) {")
pprint(ccode(eta.subs(t, 0)).replace('x.x','x').replace('x.y','y'))
print("}\n\nsourceTerm(x, y, t) {")
pprint(ccode(S).replace('x.x','x').replace('x.y','y'))
#print("}\nlaplacian(x,y,t) {")
#pprint(ccode(laplacian).replace('x.x','x').replace('x.y','y'))
print("}\ndedt(x,y,t) {")
pprint(ccode(dedt).replace('x.x','x').replace('x.y','y'))
print("}\n")

'''
dadx   = A1*B1*t*cos(x*B1) + A2*B2*cos(x*B2 + C2*t)
d2adx2 = -A1*B1**2*t*sin(x*B1) - A2*B2**2*sin(x*B2 + C2*t)
alpha  = A1*t*sin(x*B1) + A2*sin(x*B2 + C2*t) + 0.25
dadt   = A1*sin(x*B1) + A2*C2*cos(x*B2 + C2*t)
Z      = 0.5*sqrt(2)*(-y + alpha)/sqrt(kappa)

Sdw = (1-tanh(Z)**2)/sqrt(16*kappa)*(sqrt(2)*dadt
                                     - 2*sqrt(kappa) * tanh(Z) * dadx**2
                                     - sqrt(2)*kappa*d2adx2
)
Spy = (1-tanh(Z)**2)/sqrt(16*kappa)*(sqrt(2)*(dadt)
                                     + 2*sqrt(kappa)*dadx**2*tanh(Z)
                                     + sqrt(2)*kappa*d2adx2
)
'''
