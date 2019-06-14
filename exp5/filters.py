import numpy as np

class LMS:
    """
    Implementation of LMS adaptive filter.
    """
    def __init__(self, M=5, mu=.03):
        """
        Create an LMS adaptive filter.

        Parameters
        ----------
        M : int, optional
            Filter dimension. If not given, will be set to standard size of 5
            coefficients.
        mu : float, optional
            Adaptive filter step. If not given, will be set to standard of .03.
            Larger step size will possibly result in fastest convergence but less
            precision. If too large will lead to divergence.
        
        Returns
        -------
        out : LMS object
            Implementation of LMS adaptive filter.
        """
        self.M = M
        self.mu = mu
        
        self.x = np.array([]) #buffer x
        
        self.W = np.array([np.zeros(self.M)] * (self.M - 1)) #filter coefficients - array of arrays of M coefficients
        self.y_ = np.zeros(self.M - 1) #filter output
        self.e = np.array([]) #filter error
        
    def buffer(self, x, d):
        """
        Filter input. x[n] and d[n] values are passed to the filter through this 
        method.
        The filter will automatically compute its output and update w values 
        after the first M samples of x and d are received. After that, the filter
        will be updated after each new input.

        Parameters
        ----------
        x : float
            One sample of x[n] signal.
        d : float
            One sample of d[n] signal.
        
        Returns
        -------
        None
        """
        self.x = np.append(self.x, x)
        if(len(self.x) == self.M):
            self.update(d)
        else:
            self.e = np.append(self.e, d - self.y_[-1])
            
    def update(self, d):
        """
        Update the filter coefficients and calculate its output. This method is
        automatically called on the correct moments by the 'buffer' routine.
        
        Parameters
        ----------
        d : float
            Current sample of d[n] signal.
        
        Returns
        -------
        None
        """
        X = np.flip(self.x, axis=0) #X[n]
        
        self.y_ = np.append(self.y_, np.dot(X, self.W[-1])) #y_[n]
        self.e = np.append(self.e, d - self.y_[-1]) #e[n]
        
        W_new = [self.W[-1] + self.mu * self.e[-1] * X] #W[n+1]
        self.W = np.concatenate((self.W, W_new)) 
        
        self.x = np.delete(self.x, 0) #delete oldest buffer element


class NLMS:
    """
    Implementation of Normalized LMS adaptive filter.
    """
    def __init__(self, M=5, mu_0=1, epsilon=.001):
        """
        Create an NLMS adaptive filter.

        Parameters
        ----------
        M : int, optional
            Filter dimension. If not given, will be set to standard size of 5
            coefficients.
        mu_0 : float, optional
            Adaptive filter step numerator. If not given, will be set to 
            standard of 1.
            Should be 0 < mu_0 < 2 for guaranteed stability.
        
        Returns
        -------
        out : NLMS object
            Implementation of NLMS adaptive filter.
        """
        self.M = M
        self.mu_0 = mu_0
        self.epsilon = epsilon
        
        self.x = np.array([]) #buffer x
        
        self.W = np.array([np.zeros(self.M)] * (self.M - 1)) #filter coefficients - array of arrays of M coefficients
        self.y_ = np.zeros(self.M - 1) #filter output
        self.e = np.array([]) #filter error
        
    def buffer(self, x, d):
        """
        Filter input. x[n] and d[n] values are passed to the filter through this 
        method.
        The filter will automatically compute its output and update w values 
        after the first M samples of x and d are received. After that, the filter
        will be updated after each new input.

        Parameters
        ----------
        x : float
            One sample of x[n] signal.
        d : float
            One sample of d[n] signal.
        
        Returns
        -------
        None
        """
        self.x = np.append(self.x, x)
        if(len(self.x) == self.M):
            self.update(d)
        else:
            self.e = np.append(self.e, d - self.y_[-1])
            
    def update(self, d):
        """
        Update the filter coefficients and calculate its output. This method is
        automatically called on the correct moments by the 'buffer' routine.
        
        Parameters
        ----------
        d : float
            Current sample of d[n] signal.
        
        Returns
        -------
        None
        """
        X = np.flip(self.x, axis=0) #X[n]
        
        self.y_ = np.append(self.y_, np.dot(X, self.W[-1])) #y_[n]
        self.e = np.append(self.e, d - self.y_[-1]) #e[n]
        
        W_new = [self.W[-1] + self.mu_0 / ((X ** 2).sum() + self.epsilon) * self.e[-1] * X] #W[n+1]
        self.W = np.concatenate((self.W, W_new)) 
        
        self.x = np.delete(self.x, 0) #delete oldest buffer element
