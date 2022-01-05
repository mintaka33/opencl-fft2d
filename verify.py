import os
import numpy as np

app_dir = 'fft2d\\x64\\Debug'

def execute_cmd(cmd):
    print('#'*8, cmd)
    os.system(cmd)

def gpu_fft(w, h):
    cmd = 'cd %s && fft2d.exe %d %d %s' % (app_dir, w, h, '-d')
    execute_cmd(cmd)
    result = np.genfromtxt('%s\\result.txt'%app_dir, dtype=float, delimiter=",")
    # r, i = result[:, 0::2], result[:, 1::2]
    return result[:, :-1]

def numpy_fft(w, h):
    a = np.arange(h*w).reshape((h, w)).astype(np.float32)
    # a = np.random.random((h, w))
    b = np.fft.fft2(a)
    result = np.zeros((h, w*2)).astype(np.float32)
    result[:, 0::2] = b.real
    result[:, 1::2] = b.imag
    return result

def dump_result(data, tag):
    filename = 'dump_%s_%dx%d.txt' % (tag, data.shape[1], data.shape[0])
    np.savetxt(filename, data, fmt='%-14f', delimiter=', ')

for h in range(4, 100, 17):
    for w in range(4, 100, 17):
        gpu = gpu_fft(w, h)
        dump_result(gpu, 'gpu')
        ref = numpy_fft(w, h)
        dump_result(ref, 'ref')
        print('INFO: [%dx%d] sum of delta = %f, max = %f' % (w, h, np.sum(np.abs(ref - gpu)), np.max(np.abs(ref - gpu))))

print('done')