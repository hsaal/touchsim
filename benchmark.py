import touchsim as ts
import numpy as np
import time

pins = [1, 3, 10, 30, 100, 300]
recs = [1, 3, 10, 30, 100, 300, 1000]
#sf = [100., 300., 1000., 3000., 10000.]
sf = [5000.]

lp = len(pins)
lr = len(recs)
lsf = len(sf)

x,y = np.meshgrid(np.linspace(-5.,5.,100),np.linspace(-5.,5.,100))
x = np.reshape(x,(-1,1))
y = np.reshape(y,(-1,1))

t = np.zeros((lp,lr,4))
tottime = np.zeros((lp,lr,lsf))

for kk in range(lsf):
    for ii in range(lp):
        for jj in range(lr):
            npin = pins[ii]
            nrec = recs[jj]
            trace = np.random.randn(npin,int(sf[kk]))

            timall = time.time()
            s = ts.Stimulus(trace=trace,
                location=np.concatenate((x[0:npin],y[0:npin]),axis=1),fs=sf[kk])
            t[ii,jj,0] = time.time()-timall

            timaffpop = time.time()
            ap = ts.AfferentPopulation(affclass='RA',
                location=np.concatenate((x[0:nrec],y[0:nrec]),axis=1))
            t[ii,jj,1] = time.time() - timaffpop

            timresp = time.time()
            rc = ap.response(s)
            t[ii,jj,2] = time.time() - timresp
            t[ii,jj,3] = time.time() - timall

tottime[:,:,kk] = t[:,:,3]

np.set_printoptions(precision=3,suppress=True)
print(tottime[:,:,0])
