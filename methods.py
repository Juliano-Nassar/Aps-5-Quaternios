'''
referências Numpy:
- numpy.array
    - https://numpy.org/doc/stable/reference/generated/numpy.array.html
- np.array2string
    - https://numpy.org/doc/stable/reference/generated/numpy.array2string.html
- numpy.dot
    - https://numpy.org/doc/stable/reference/generated/numpy.dot.html
'''

import numpy as np

# Vector class
class Vector:
    def __init__(self,vector):  
        self.v = np.array(vector)
        
    def __mul__(self, v2):
        v = v2.v
        result = self.v.dot(v)
        return Vector(result)
    
    def __rmul__(self, n):
        result = n*self.v
        return Vector(result)
    
    def __add__(self,v2):
        result = self.v + v2.v
        return Vector(result)
    
    def __sub__(self,v2):
        result = self.v - v2.v
        return Vector(result)
    
    def length(self):
        return np.sqrt((self.v**2).sum())
    
    def normalized(self):
        result = self.v/self.length()
        return Vector(result)
    
    def T(self):
        return Vector(self.v.T)
    
    def __str__(self):
        v_string = np.array2string(self.v, formatter={'float_kind':lambda x: "%.3f" % x})
        
        if len(self.v.shape)>1:
            v_string = v_string[1:-1]
            
        v_string =  ' '+v_string
        
        return v_string.replace('[','|').replace(']','|')
    
# Euler Vector subclass
class Euler_Vector(Vector):
    def __init__(self,angle,dimension=3,axis = 'x'):
        c = np.cos(angle)
        s = np.sin(angle)
        if dimension == 3:
            if axis =='x' or axis == 0:
                vector = [[1,0, 0],
                          [0,c,-s],
                          [0,s, c]]
            elif axis =='y' or axis == 1:
                vector = [[ c,0,s],
                          [ 0,1,0],
                          [-s,0,c]]
            elif axis =='z' or axis == 2:
                vector = [[c,-s,0],
                          [s, c,0],
                          [0, 0,1]]
        elif dimension == 2:
            vector = [[c,-s],
                      [s, c]]
        else:
            raise("Dimensão inválida, utilizar ou 2 ou 3 dimensões")

        Vector.__init__(self,vector)
        
# Quaternio Class    
class Quaternio:
    def __init__(self,quat = [0,0,0,1], angle = None, axis = None, agg = []):
        
        # quaternio = [r, i, j, k]
        # axis = [ux, uy, uz]
        # agg = list of all transformations applied for traceback (Not implemented)
        
        quaternio = quat.copy()
        
        # if rotation specified
        if angle is not None and axis is not None:
            self.axis = axis.copy()
            self.axis.insert(0,1)
            self.axis = np.array(self.axis)
            
            s = np.sin(angle/2)
            c = np.cos(angle/2)
            
            quaternio = np.array([c,s,s,s])
            quaternio = quaternio*self.axis
            
        if len(agg) == 0:
            agg.append(quaternio)
            
        self.q = np.array(quaternio)
        self.agg = agg
        
    # Multiply method taken from:
    # https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    def __mul__(self, q2):
        p = self.q
        q = q2.q
        
        w1, x1, y1, z1 = p
        w0, x0, y0, z0 = q

        result = [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                  x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                  -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                  x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]
        
        return Quaternio(result, agg = self.agg+q2.agg)
    
    def __rmul__(self, n):
        result = n*self.q
        return Quaternio(result, agg = self.agg+[n])
    
    def __add__(self,q2):
        result = self.q + q2.q
        return Quaternio(result, agg = self.agg+q2.agg)
    
    def __sub__(self,q2):
        result = self.q - q2.q
        return Quaternio(result, agg = self.agg+q2.agg)
    
    def __str__(self):
        a = self.q[0]
        i = self.q[1]
        j = self.q[2]
        k = self.q[3] 
        return f"{a} + {i}i + {j}j + {k}k"
    
    def length(self):
        return np.sqrt((self.q**2).sum())
    
    def normalized(self):
        result = self.q/self.length()
        return Quaternio(result, agg = self.agg+[self.length()])
    
    def rotation_matrix(self):
        norm = self.normalized()
        q_ref = norm.q
        
        qr = q_ref[0]
        qx = q_ref[1]
        qy = q_ref[2]
        qz = q_ref[3]
        
        R = [[1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qr), 2*(qx*qz + qy*qr), 0],
             [2*(qx*qy + qz*qr), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qr), 0],
             [2*(qx*qz - qy*qr), 2*(qy*qz + qx*qr), 1 - 2*(qx**2 + qy**2), 0],
             [0,                 0,                 0,                     1]]
        
        return Vector(R)