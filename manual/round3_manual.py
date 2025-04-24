import math
import numpy as np
def integral_fR(a, b):
    # first part
    overlap1 = max(0, min(b, 200) - max(a, 160))
    part1 = (1 / 80) * overlap1

    # part 2
    overlap2 = max(0, min(b, 320) - max(a, 250))
    part2 = (1 / 140) * overlap2

    return part1 + part2


def objective(low, high):
    """Compute the value of the surrogate objective function.

    Parameters
    ----------
    low : int
        Value of the low bid.
    high : int
        Value of the high bid.
    
    Returns
    -------
    float
        Value of the surrogate objective function.
    """
    lhs = (320-low) * integral_fR(160,low)
    rhs = (320-high) *integral_fR(low,high)
    return (lhs + rhs)

argmax = []
val_max = 0
for low in range(160, 321):
    for high in range(low, 321):
        comp = objective(low, high)
        if math.isclose(comp, val_max):
            argmax.append((low, high))
        elif comp > val_max:
            val_max = comp
            argmax = [(low, high)]
if len(argmax) > 1:
    print('Maximizers:', argmax)
else:
    print('Maximizer:', argmax[0])


#### after this is the section that alters based on te 

def solve(p_avg):
    """Given the average of second bids, find the optimal low and high bids.

    Parameters
    ----------
    p_avg : float
        Average value of second bids.
    
    Returns
    -------
    argmax : list of tuple
        Maximizers.
    val_max : float
        Maximal profit.
    """
    val_max = float('-inf')
    argmax = []
    for l in range(160, 320):
        for h in range(l, 320):
            temp = (320 - l) *integral_fR(160,l)
            temp2 = (320 - h) *integral_fR(l,h)
            val = temp + temp2 * (1 if p_avg <= h else ((320-p_avg)/(320-h))**3)
            if math.isclose(val, val_max):
                argmax.append((l, h, val))
            if val > val_max:
                val_max = val
                argmax = [(l, h)]
    return argmax, val_max

for p_avg in np.arange(270,320,0.5):
    argmax, val_max = solve(p_avg)
    if len(argmax) > 1:
        print("p_avg:", p_avg, "  Maximizers:", argmax, " Profit:", f"{val_max:.2f}")
    else:
        print("p_avg:", p_avg, "  Maximizer:", argmax[0], " Profit:", f"{val_max:.2f}")