import numpy as np

def inte_basis(i, u, nodevec):
    if (nodevec[i] <= u) and (u < nodevec[i+1]):
        result = (u-nodevec[i])**3/(3*(nodevec[i+2]-nodevec[i])*(nodevec[i+1]-nodevec[i]))
    elif (nodevec[i+1] <= u) and (u < nodevec[i+2]):
        result = (nodevec[i+1]-nodevec[i])**2/(3*(nodevec[i+2]-nodevec[i])) + (-u**3/3+u**2*(nodevec[i+2]+nodevec[i])/2-nodevec[i+2]*nodevec[i]*u+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+2]+nodevec[i])/2+nodevec[i+2]*nodevec[i]*nodevec[i+1])/((nodevec[i+2]-nodevec[i])*(nodevec[i+2]-nodevec[i+1])) + (-u**3/3+u**2*(nodevec[i+3]+nodevec[i+1])/2-nodevec[i+3]*nodevec[i+1]*u+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+3]+nodevec[i+1])/2+nodevec[i+3]*nodevec[i+1]*nodevec[i+1])/((nodevec[i+3]-nodevec[i+1])*(nodevec[i+2]-nodevec[i+1]))
    elif (nodevec[i+2] <= u) and (u < nodevec[i+3]):
        result = (nodevec[i+1]-nodevec[i])**2/(3*(nodevec[i+2]-nodevec[i])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+2]+nodevec[i])/2-nodevec[i+2]*nodevec[i]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+2]+nodevec[i])/2+nodevec[i+2]*nodevec[i]*nodevec[i+1])/((nodevec[i+2]-nodevec[i])*(nodevec[i+2]-nodevec[i+1])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+3]+nodevec[i+1])/2-nodevec[i+3]*nodevec[i+1]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+3]+nodevec[i+1])/2+nodevec[i+3]*nodevec[i+1]*nodevec[i+1])/((nodevec[i+3]-nodevec[i+1])*(nodevec[i+2]-nodevec[i+1])) + ((u-nodevec[i+3])**3+(nodevec[i+3]-nodevec[i+2])**3)/(3*(nodevec[i+3]-nodevec[i+1])*(nodevec[i+3]-nodevec[i+2]))
    elif (nodevec[i+3] <= u):
        result = (nodevec[i+1]-nodevec[i])**2/(3*(nodevec[i+2]-nodevec[i])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+2]+nodevec[i])/2-nodevec[i+2]*nodevec[i]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+2]+nodevec[i])/2+nodevec[i+2]*nodevec[i]*nodevec[i+1])/((nodevec[i+2]-nodevec[i])*(nodevec[i+2]-nodevec[i+1])) + (-nodevec[i+2]**3/3+nodevec[i+2]**2*(nodevec[i+3]+nodevec[i+1])/2-nodevec[i+3]*nodevec[i+1]*nodevec[i+2]+nodevec[i+1]**3/3-nodevec[i+1]**2*(nodevec[i+3]+nodevec[i+1])/2+nodevec[i+3]*nodevec[i+1]*nodevec[i+1])/((nodevec[i+3]-nodevec[i+1])*(nodevec[i+2]-nodevec[i+1])) + (nodevec[i+3]-nodevec[i+2])**3/(3*(nodevec[i+3]-nodevec[i+1])*(nodevec[i+3]-nodevec[i+2]))
    else:
        result = 0
    return result


def Ic_inte_basis(j, m, u, nodevec):
    if (j==0):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = 3*((nodevec[1]-nodevec[0])**3-(nodevec[1]-u)**3)/(3*(nodevec[1]-nodevec[0])**2)/(m+1)
        else:
            result = 3*(nodevec[1]-nodevec[0])/3/(m+1)
        return result
    elif (j==1):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = 3/3*(((-u**3/3+(u**2*(nodevec[0]+nodevec[2])/2-nodevec[0]*nodevec[2]*u))-(-nodevec[0]**3/3+(nodevec[0]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]**2*nodevec[2])))/((nodevec[1]-nodevec[0])*(nodevec[2]-nodevec[1])))/(m+1)
        elif (nodevec[1]<=u) and (u<nodevec[2]):
            result = 3/3*(((-nodevec[1]**3/3+(nodevec[1]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]*nodevec[2]*nodevec[1]))-(-nodevec[0]**3/3+(nodevec[0]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]**2*nodevec[2])))/((nodevec[1]-nodevec[0])*(nodevec[2]-nodevec[1]))+((nodevec[2]-nodevec[1])**3-(nodevec[2]-u)**3)/(3*(nodevec[2]-nodevec[1])**2))/(m+1)
        else:
            result = 3/3*(((-nodevec[1]**3/3+(nodevec[1]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]*nodevec[2]*nodevec[1]))-(-nodevec[0]**3/3+(nodevec[0]**2*(nodevec[0]+nodevec[2])/2-nodevec[0]**2*nodevec[2])))/((nodevec[1]-nodevec[0])*(nodevec[2]-nodevec[1]))+(nodevec[2]-nodevec[1])/3)/(m+1)
        return result
    elif (j==2):
        if (nodevec[m-1]<=u) and (u<nodevec[m]):
            result = 3/3*((u-nodevec[m-1])**3/(3*(nodevec[m]-nodevec[m-1])**2))/2
        elif (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = 3/3*((nodevec[m]-nodevec[m-1])/3+((-u**3/3+(u**2*(nodevec[m-1]+nodevec[m+1])/2-nodevec[m-1]*nodevec[m+1]*u))-(-nodevec[m]**3/3+(nodevec[m]**2*(nodevec[m-1]+nodevec[m+1])/2-nodevec[m-1]*nodevec[m+1]*nodevec[m])))/((nodevec[m]-nodevec[m-1])*(nodevec[m+1]-nodevec[m])))/2
        else:
            result = 0
        return result
    elif (j==3):
        if (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = 3*(u-nodevec[m])**3/(3*(nodevec[m+1]-nodevec[m])**2)
        else:
            result = 0
        return result


def I_spline(m, u, nodevec):
    B_p = []
    for i in range(m-1):
        B_p.append(inte_basis(i, u, nodevec))
    for j in range(4):
        B_p.append(Ic_inte_basis(j, m, u, nodevec))
    B_p = np.array(B_p, dtype='float32')
    return B_p


def I_U(m, U, nodevec):
    I_u = np.zeros(shape=(len(U),m+3))
    for b in range(len(U)):
        u = U[b]
        I_u[b] = I_spline(m, u, nodevec)
    I_u = np.array(I_u, dtype='float32')
    return I_u



def I_S(m, c0, U, nodevec):
    B_value = []
    for b in range(len(U)):
        u = U[b]
        B_value.append(np.sum(I_spline(m, u, nodevec)*c0)) 
    B_value = np.array(B_value, dtype='float32')
    return B_value
