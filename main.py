from ihs_bt import CL
import scipy.misc as scm
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0'

if __name__ == "__main__":
    pan = scm.imread('assets/scale/taipei_pan.jpg')
    mul =  scm.imread('assets/scale/taipei_mul.jpg')

    example = CL()
    example.loadProgram("program.cl")
    example.loadData(pan, mul)
    example.execute()

    output_img = example.output()
    scm.imsave("output.jpg", output_img)
