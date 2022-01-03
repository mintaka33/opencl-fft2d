import numpy as np

a = np.arange(256).reshape((16, 16))
# a = np.ones_like(a)
np.savetxt('a.txt', a, fmt='%6.2f', delimiter=', ')

b = np.fft.fft2(a)
np.savetxt('b.txt', b, fmt='%10.2f', delimiter=', ')

print('done')