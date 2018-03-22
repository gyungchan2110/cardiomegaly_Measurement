from scipy.interpolate import UnivariateSpline
import numpy as np

def curvature_splines(x, y=None, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )

         In the second case the curve is represented as a np.array
         of complex numbers.

    error : float
        The admisible error when interpolating the splines

    Returns
    -------
    curvature: numpy.array shape (n_points, )

    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)

    xˈˈ = fx.derivative(2)(t)

    yˈ = fy.derivative(1)(t)
    #print (yˈ )
    yˈˈ = fy.derivative(2)(t)
    #print (yˈˈ )
    curvature = (0 - yˈ* xˈˈ) / np.power(xˈ** 2 + 1, 3 / 2)
    return curvature



def curvature(x, y=None, error=1):

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)
    print(std)
    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = np.gradient(x)
    print (xˈ )
    xˈˈ = np.gradient(xˈ)
    print (xˈˈ )
    yˈ = fy.derivative(1)(t)
    #print (yˈ )
    yˈˈ = fy.derivative(2)(t)
    #print (yˈˈ )
    curvature = (0 -  xˈˈ) / np.power(xˈ** 2 + 1, 3 / 2)
    return curvature