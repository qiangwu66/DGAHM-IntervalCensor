import numpy as np
def B_spline_basis(i, p, u, nodevec):
    if p == 0:
        if (nodevec[i] <= u) and (u < nodevec[i + 1]):
            result = 1
        else:
            result = 0
    else:
        length1 = nodevec[i + p] - nodevec[i]
        length2 = nodevec[i + p + 1] - nodevec[i + 1]
        if length1 == 0:
            alpha = 0
        else:
            alpha = (u - nodevec[i]) / length1
        if length2 == 0:
            beta = 0
        else:
            beta = (nodevec[i + p + 1] - u) / length2
        result = alpha * B_spline_basis(i, p - 1, u, nodevec) + beta * B_spline_basis(i + 1, p - 1, u, nodevec)
    return result


def B_other_basis(j, m, u, nodevec):
    if (j==0):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = (nodevec[1]-u)**2/(nodevec[1]-nodevec[0])**2
        else:
            result = 0
        return result
    elif (j==1):
        if (nodevec[0]<=u) and (u<nodevec[1]):
            result = (nodevec[2]-u)*(u-nodevec[0])/((nodevec[2]-nodevec[0])*(nodevec[1]-nodevec[0]))
        elif (nodevec[1]<=u) and (u<nodevec[2]):
            result = (nodevec[2]-u)**2/((nodevec[2]-nodevec[0])*(nodevec[2]-nodevec[1]))
        else:
            result = 0
        return result
    elif (j==2):
        if (nodevec[m-1]<=u) and (u<nodevec[m]):
            result = (u-nodevec[m-1])**2/((nodevec[m+1]-nodevec[m-1])*(nodevec[m]-nodevec[m-1]))
        elif (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = (u-nodevec[m-1])*(nodevec[m+1]-u)/((nodevec[m+1]-nodevec[m-1])*(nodevec[m+1]-nodevec[m]))
        else:
            result = 0
        return result
    elif (j==3):
        if (nodevec[m]<=u) and (u<=nodevec[m+1]):
            result = (u-nodevec[m])**2/(nodevec[m+1]-nodevec[m])**2
        else:
            result = 0
        return result


def B_spline(m, u, nodevec):
    B_p = [] 
    for i in range(m-1):
        B_p.append(B_spline_basis(i, 2, u, nodevec))
    for j in range(4):
        B_p.append(B_other_basis(j, m, u, nodevec))
    B_p = np.array(B_p, dtype='float32')
    return B_p


def B_S2(m, U, nodevec):
    B_u = np.zeros(shape=(len(U),m+3))
    for b in range(len(U)):
        u = U[b]
        B_u[b] = B_spline(m, u, nodevec)
    B_u = np.array(B_u, dtype='float32')
    return B_u
