import numpy as np
import matplotlib.pyplot as plt

push = np.array([[1,0],[1,0]])
pop = np.array([[0,1],[0,1]])
mf = lambda pos: np.sin(pos[0])*0.5 +0.5
def m0(pos):
    m = mf(pos)
    zero = np.zeros_like(m)
    one = np.ones_like(m)
    return np.array([[m,zero],[zero,one]]).T

def W(pos, s, W0):
    sk = s[:, np.newaxis,np.newaxis]
    mk = m0(pos)
    return (sk[0].T * (push @ W0) + sk[1].T * (pop @ W0) + sk[2].T * (mk @ W0)) / (sk[0] + sk[1] + sk[2]).T

def s(pos,t,s0):
    x = pos[0]
    y = pos[1]
    s = s0
    tiled_sin_t =  np.tile(np.sin(2*np.pi*t),[len(x),len(y)])
    tiled_cos_t =  np.tile(np.cos(2*np.pi*t),[len(x),len(y)])
    return (np.array([np.sin(s[2]+0.5), np.sin(s[2]+0.4), tiled_sin_t + s[2]])) # some arbitrary function of x and t and previous state

def make_images(u, x, y, t0, s0, dt, n):
    Wc = np.tile(np.identity(2), (len(x),len(y),1,1))
    sc = np.tile(s0, (len(x),len(y),1)).T
    pos = np.array([x,y])
    outs = []
    for i in range(n):
        t = t0 + i * dt
        sc = s(pos, t, sc)
        Wc = W(pos, sc, Wc)
    # return Wc[:,:,0,0] * u
        outs.append(Wc[:,:,0,0] * u)
    return np.array(outs)
    
    
u0 = np.ones((100,100))#np.random.rand(100,100)
x, y = np.meshgrid(np.linspace(0,1,100) - 0.5, np.linspace(0,1,100) - 0.5)
s0 = np.array([0,0,0])
t0 = 0
dt = 0.1
n = 200
outs = make_images(u0, x, y, t0, s0, dt, n)
for i, im in enumerate(outs):
    plt.imshow(im)
    plt.savefig(f"testing/att_lsys/{i}.png")
# for im in outs:
#     m = plt.imshow(im)
#     plt.colorbar(m)
#     plt.show()
