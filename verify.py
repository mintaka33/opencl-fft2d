import os
import numpy as np

app_dir = 'fft2d\\x64\\Debug'

def execute_cmd(cmd):
    print('#'*8, cmd)
    os.system(cmd)

def gpu_fft(w, h):
    cmd = '%s\\fft2d.exe %d %d %s' % (app_dir, w, h, '-d')
    execute_cmd(cmd)
    result = np.genfromtxt('result.txt', dtype=np.float, delimiter=",")
    # r, i = result[:, 0::2], result[:, 1::2]
    return result[:, :-1]

def numpy_fft(w, h):
    a = np.arange(h*w).reshape((h, w))
    b = np.fft.fft2(a)
    result = np.zeros((h, w*2))
    result[:, 0::2] = b.real
    result[:, 1::2] = b.imag
    return result

for h in range(16, 33, 16):
    for w in range(16, 33, 16):
        gpu = gpu_fft(w, h)
        print('gpu shape = ', gpu.shape)
        ref = numpy_fft(w, h)
        print('ref shape = ', ref.shape)
        print('INFO: [%dx%d] sum of delta = %f' % (w, h, np.sum(np.abs(ref - gpu))))

print('done')