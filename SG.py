from cProfile import label
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

#N=401;d1=30*10^-9;#m  default30d2=20*10^-9;#defalut20
hbar=6.582119569e-16#eVs

m0=9e-31;#kg
deltax=1e-10#m

Eg1=1.98;me1=0.1*m0;mh1=-0.63*m0;
Eg2=1.42;me2=0.067*m0;mh2=-0.51*m0;
deltaEc=0.28;

class materialGaAs:
    def __init__(self):
        self.deltaEso = 0.341*const.e
        self.Eg = 1.519*const.e + self.deltaEso/3.
        self.Ep = 28.8*const.e 
        self.m = 0.067*const.m_e
        self.Ev0   = - 0.8*const.e


        self.epsr_static = 13.1
        self.E_LO = 36.1e-3*const.e
        self.nr = 3.3
        self.eps_r_0 = 13.1
        self.eps_r_inf = self.nr**2

class materialAlAs:
    def __init__(self):
        self.deltaEso = 0.28*const.e # "eV";
        self.Eg = 3.099*const.e + self.deltaEso/3.
        self.Ep = 21.1*const.e # "eV";
        self.m = 0.15*const.m_e
        self.Ev0   = - 1.33*const.e 


class materialInAs:
    def __init__(self):
        self.deltaEso = 0.39*const.e # "eV";
        self.Eg = 0.417*const.e #"J"
        self.Ep = 21.5*const.e # "eV";
        self.m = 0.026*const.m_e
        self.Ev0   = - 0.59*const.e


#Eg(A_1-x B_x) = (1-x)E_g(A) + x E_g(B)
class materialAlGaAs:
    def __init__(self,x):
        A = materialAlAs()
        B = materialGaAs()

        for key in A.__dict__:
            self.__dict__[key] =  x*A.__dict__[key] +  (1-x)*B.__dict__[key]




class material:
    def __init__(self,Eg,Ec,me,mh):
        self.Eg=Eg
        self.Ec=Ec
        self.m=me
        self.mh=mh

GaAs=material(Eg=Eg1,Ec=Eg1,me=me1,mh=me1) 
AlGaAs=material(Eg=Eg2,Ec=Eg1-deltaEc,me=me2,mh=me2)


class device:
    def __init__(self):
        self.me=np.zeros(0)
        self.V=np.zeros(0)
        self.thikness=0
    def attachlayer(self,material,thikness):
        e=int(thikness/deltax)
        self.me=np.append(self.me,np.ones(e)*material.m)
        self.V=np.append(self.V,np.ones(e)*(material.Eg + material.Ev0)/const.e)
        self.thikness+=thikness

    def solve(self):
        a=(hbar**2/(2*self.me*(deltax**2)))/(6.242*1e18)
        h=np.zeros(len(a))

        for i,ai in enumerate(a):
            if(i==0):
                continue
            h[i]=a[i-1]+a[i]+self.V[i]
        
        a = a[1:]#bei a brauch man wedet das nullte no das m te element
        a = a[:-1]
        h = h[1:]#bei h braucht man das 0 te nicht

        H = np.diag(-a, 1)
        H += np.diag(h, 0)
        H += np.diag(-a, -1)
 

        eigenvalues,eigenvectors = np.linalg.eigh(H)

        self.eigenvalues=eigenvalues
        self.eigenvectors=eigenvectors

        return eigenvalues,eigenvectors

    def plot(self):
        x=np.arange(deltax,self.thikness,deltax) 
        trans_frq="f[Hz]:"
        trans_frq+=	"{:.2e}".format((self.eigenvalues[1]-self.eigenvalues[0])/(hbar*2*np.pi))
        trans_frq+=",E[eVs]:"
        trans_frq+=	"{:.2e}".format((self.eigenvalues[1]-self.eigenvalues[0]))
        


        for i in range(3):
            plt.plot(np.linspace(0,self.thikness,len(self.eigenvectors[:,i])),-self.eigenvectors[:,i]+self.eigenvalues[i])
            plt.plot(np.linspace(0,self.thikness,len(self.eigenvectors[:,i])),self.eigenvectors[:,i]**2+self.eigenvalues[i])

        plt.plot(np.linspace(0,self.thikness,len(self.V)),self.V)
        #plt.title(trans_frq)
        plt.legend()
        plt.ylabel('Energie in [eV]') 
        plt.xlabel('länge in [m]') 
        plt.savefig("well.png")
        plt.show()


if True:
    Al=0.3
    well=device()
    well.attachlayer(materialAlGaAs(Al),10e-9)
    well.attachlayer(materialGaAs(),7.5e-9)
    well.attachlayer(materialAlGaAs(Al),2e-9)
    well.attachlayer(materialGaAs(),11.8e-9)  
    well.attachlayer(materialAlGaAs(Al),10e-9)
    delta_V=0.001
    for i,Vi in enumerate(well.V):
        well.V[i]+=i*delta_V


    eigenvalues,eigenvectors=well.solve()
    eigenvectors/=np.sqrt(deltax)

    z=np.arange(0,well.thikness,deltax)
    print("z21",np.trapz(eigenvectors[:,2]*eigenvectors[:,1],dx=deltax))
    print("z11",np.trapz(eigenvectors[:,1]*eigenvectors[:,1],dx=deltax))
    print("z22",np.trapz(eigenvectors[:,2]*eigenvectors[:,2],dx=deltax))
    print("z31",np.trapz(eigenvectors[:,3]*eigenvectors[:,1],dx=deltax))

    eigenvectors*=np.sqrt(deltax)
    well.plot()
    eigenvectors/=np.sqrt(deltax)




if True:#self consistent part
    import numpy as np
    import scipy.integrate as integrate


    fermi = 0.7
    T = 300
    kB = 8.6173303e-5


    def fermi_integral(E, fermi, T):
        return 1 / (1 + np.exp((E - fermi) / (kB * T)))

    result = integrate.quad(fermi_integral, eigenvalues[0], np.inf  ,args=( fermi, T))
    print(result)
    roh=-result[0]*well.me[1:]/(np.pi*hbar**2)*eigenvectors[:,0]**2*const.e*5e10
    #roh=-eigenvectors[:,0]**2*1e10*const.e*(deltax**2)/(12.9*1.5*1e-10)

    fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, sharex=True)
    fig.suptitle('Aligning x-axis using sharex')

    well.eigenvectors*=np.sqrt(deltax)
    for i in range(3):
        ax1.plot(np.linspace(0,well.thikness,len(well.eigenvectors[:,i]))/deltax,well.eigenvectors[:,i]**2+well.eigenvalues[i])
    ax1.plot(np.linspace(0,well.thikness,len(well.V))/deltax,well.V)
    ax2.plot(range(len(roh)),roh,label="roh")
    well.eigenvectors/=np.sqrt(deltax)

    h1=np.ones(len(roh))*-2
    h2=np.ones(len(roh)-1)
    A = np.diag(h1, 0)
    A += np.diag(h2, 1)
    A += np.diag(h2, -1)

    from scipy.linalg import solve
    s = solve(A, roh)
    ax3.plot(range(len(s)),s,label="pot")
    ax4.plot(range(len(s)),s+well.V[1:],label="pot")
    ax4.plot(range(len(s)),well.V[1:],label="ori")
    plt.legend()
    plt.show()
    #print(roh)

if False:
    def func3(x, y):
        well=device()
        Al=0.05
        well.attachlayer(materialAlGaAs(Al),10e-9)
        well.attachlayer(materialGaAs(),x)
        well.attachlayer(materialAlGaAs(Al),2e-9)
        well.attachlayer(materialGaAs(),y)
        well.attachlayer(materialAlGaAs(Al),10e-9)


        eigenvalues,eigenvectors=well.solve()

        eigenvectors/=np.sqrt(deltax)

        #z=np.arange(0,well.thikness,deltax)
        z=np.arange(0,len(eigenvectors[:,2]),1)*deltax
        return np.trapz(eigenvectors[:,0]*z*eigenvectors[:,1],dx=deltax)   #select between heatmap of matix element and lasingfrequenz 
        #return eigenvalues[1]-eigenvalues[0]


    dx=dy = 1e-9

    x = np.arange(6e-9, 12e-9, dx)
    y = np.arange(6e-9, 12e-9, dy)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    X, Y = np.meshgrid(x, y)


    E=np.zeros((len(x),len(y)))
    for ii,i in enumerate(x):
        for jj,j in enumerate(y):
            E[ii,jj] = func3(i, j)
    
    plt.contourf(x, y, E)
    plt.colorbar()

    plt.show()
        #plt.close()


if False:
    squared=device()
    squared.attachlayer(GaAs,20e-9)
    squared.attachlayer(GaAs,20e-9)
    squared.attachlayer(GaAs,20e-9)
    for i,V in enumerate(squared.V):
        if i>200 and i<400:
            squared.V[i]=1e15*(i*deltax-squared.thikness/2)**2-0.6
        else:
            squared.V[i]=0.0
    print(squared.V)
    squared.solve()
    plt.plot(np.arange(deltax,len(squared.V)*deltax,deltax),squared.eigenvectors[:,0]**2+squared.eigenvalues[0])
    plt.plot(np.arange(deltax,len(squared.V)*deltax,deltax),squared.eigenvectors[:,1]**2+squared.eigenvalues[1],label=squared.eigenvalues[1]-squared.eigenvalues[0])
    plt.plot(np.arange(deltax,len(squared.V)*deltax,deltax),squared.eigenvectors[:,2]**2+squared.eigenvalues[2])
    plt.plot(np.arange(deltax,len(squared.V)*deltax,deltax),squared.V[1:])






















if False:
    well.solve()
    well.plot()


    plt.legend()
    plt.ylabel('Energie in [eV]') 
    plt.xlabel('länge in [m]') 
    plt.show()

if False:#lösung nicht mehr ändern
    well=device()
    well.attachlayer(GaAs,10e-9)
    well.attachlayer(AlGaAs,1.6e-9)
    well.attachlayer(GaAs,0.4e-9)
    well.attachlayer(AlGaAs,3.5e-9)
    well.attachlayer(GaAs,10e-9)
    well.solve()
    well.plot()