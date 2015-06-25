#Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy
import time

class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #print(fstr)
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def loadData(self, pan, mul):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays
        self.shape = pan.shape

        self.pan = pan.ravel().astype(numpy.int32)
        self.mul = mul
        self.pan2 = pan
        self.k = 0.5

        self.r = mul[:, :, 0].ravel().astype(numpy.int32)
        self.g = mul[:, :, 1].ravel().astype(numpy.int32)
        self.b = mul[:, :, 2].ravel().astype(numpy.int32)

        time_start = time.time()
        ###
        #create OpenCL buffers
        self.r_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.r)
        self.g_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.g)
        self.b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.pan_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pan)
        self.dest_r_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.r.nbytes)
        self.dest_g_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.g.nbytes)
        self.dest_b_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.b.nbytes)
        ###
        finish_time = time.time() - time_start
        print("H->D finish time:", finish_time, " s")
    def nonparallize(self):
        r = self.mul[:, :, 0]
        g = self.mul[:, :, 1]
        b = self.mul[:, :, 2]
        i = (r*0.171 + g*0.2 + b*0.171) / 0.632
        kx__pan_minus_iii = self.k*(self.pan2-i)
        coe = self.pan2/(i+kx__pan_minus_iii)
        xnr  = coe * (r + kx__pan_minus_iii)
        xng  = coe * (g + kx__pan_minus_iii)
        xnb  = coe * (b + kx__pan_minus_iii)

    def execute(self):
        time_start = time.time()
        ###
        self.program.calculate(self.queue, self.pan.shape, (256,), self.r_buf, self.g_buf, self.b_buf, self.pan_buf, self.dest_r_buf, self.dest_g_buf, self.dest_b_buf).wait()
        ###
        finish_time = time.time() - time_start
        print("CPU finish time:", finish_time, " s")
    def output(self):
        nr = numpy.empty_like(self.r)
        ng = numpy.empty_like(self.g)
        nb = numpy.empty_like(self.b)

        time_start = time.time()
        ###
        cl.enqueue_read_buffer(self.queue, self.dest_r_buf, nr).wait()
        cl.enqueue_read_buffer(self.queue, self.dest_g_buf, ng).wait()
        cl.enqueue_read_buffer(self.queue, self.dest_b_buf, nb).wait()
        ###
        finish_time = time.time() - time_start
        print("D->H finish time:", finish_time, " s")

        rnr = numpy.reshape(nr, self.shape)
        rng = numpy.reshape(ng, self.shape)
        rnb = numpy.reshape(nb, self.shape)

        output_img = numpy.empty_like(self.mul)
        
        output_img[:, :, 0] = rnr
        output_img[:, :, 1] = rng
        output_img[:, :, 2] = rnb
        return output_img

