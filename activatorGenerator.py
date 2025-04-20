

from rich import print
from matplotlib import pyplot as plt





























MAX = 10.0
SPAN = 10.0
LEAK = 0.1

if LEAK >= MAX/SPAN:
    print(f'"L" needs to be lower than "MAX/SPAN"\nL        = {LEAK:.5f}\nMAX/SPAN = {MAX/SPAN:.5f}')
    quit()

gg = (MAX - LEAK*SPAN)/(.5*SPAN*SPAN)

a = LEAK
b = gg
c = LEAK
d = -gg
e = LEAK + 2 * SPAN * gg
f = -SPAN*SPAN*gg-LEAK*SPAN+MAX
g = LEAK
h = LEAK*SPAN + MAX - LEAK*SPAN*2


print(f'f1: {a:.5f}X')
print(f'f2: {b:.5f}X^2 + {c:.5f}X')
print(f'f3: {d:.5f}X^2 + {e:.5f}X + {f:.5f}')
print(f'f4: {g:.5f}X   + {h:.5f}')





code =     \
f'''
    // leaky decay line
    if (val <= 0.0f)
        return val * {a:.5f}f;

    // parabola upward
    if (val <= {SPAN/2:.5f}f)
        return {b:.5f}f * val * val + {c:.5f}f * val;
    
    // parabola downward
    if (val <= {SPAN:.5f}f)
        return ({d:.5f}f * val * val) + ({e:.5f}f * val) + {f:.5f}f;

    // leaky positive line
    return ( {g:.5f}f * val ) + {h:.5f}f;
'''

der_code = \
f'''
    // leaky decay line
    if (val <= 0.0f)
        return {LEAK:.5f}f;

    // parabola upward
    if (val <= {SPAN/2:.5f}f)
        return {2 * gg:.5f}f * val + {LEAK:.5f}f;
    
    // parabola downward
    if (val <= {SPAN:.5f}f)
        return (-{gg * 2:.5f}f * val) + {LEAK + 2 * SPAN * gg:.5f}f;

    // leaky positive line
    return {LEAK:.5f}f;
'''

while '0f' in code:
    code = code.replace('0f', 'f')
code = code.replace('.f', '.0f')

while '0f' in der_code:
    der_code = der_code.replace('0f', 'f')
der_code = der_code.replace('.f', '.0f')

print(code)
print(der_code)



def act(val):


    # leaky decay line
    if val <= 0:
        return a * val

    # parabola upward
    if val <= SPAN/2: 
        return b * val * val + c * val
    
    # parabola downward
    if val <= SPAN:
        return (d * val * val) + (e * val) + f

    # leaky positive line
    return ( g * val ) + h


def der(val):
    # leaky decay line
    if val <= 0:
        return LEAK

    # parabola upward
    if val <= SPAN/2:
        return 2 * gg * val + LEAK
    
    # parabola downward
    if val <= SPAN:
        return (-gg * 2 * val) + LEAK + 2 * SPAN * gg

    # leaky positive line
    return LEAK

aa = int(( -1  * SPAN*.1) * 10000)
bb = int((SPAN + SPAN) * 1000)


plt.plot([act(i/1000) for i in range(aa, bb)])
plt.plot([der(i/1000) for i in range(aa, bb)])
plt.show()