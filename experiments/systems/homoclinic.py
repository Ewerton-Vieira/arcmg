import numpy as np 
from arcmg.system import BaseSystem
from scipy.integrate import solve_ivp

class Homoclinic(BaseSystem):
    def __init__(self, t_span=1, integration_steps=10, **kwargs):
        super().__init__(**kwargs)
        self.name = "homoclinic"
        self.state_bounds = np.array([[-1, 1]]*2)
        self.t_span = t_span
        self.integration_steps = integration_steps

    # Define the system of differential equations
    def system(t, z):
        x, y = z
        dxdt = -(-y - (x**4 / 4 - x**2 / 2 + y**2 / 2) * (x**3 - x))
        dydt = -(x**3 - x - (x**4 / 4 - x**2 / 2 + y**2 / 2) * y)
        return [dxdt, dydt]
    # Latex form
    """
    \[
    x' = -\left( -y - \left( \frac{x^4}{4} - \frac{x^2}{2} + \frac{y^2}{2} \right)(x^3 - x) \right)
    \]
    \[
    y' = -\left( x^3 - x - \left( \frac{x^4}{4} - \frac{x^2}{2} + \frac{y^2}{2} \right) y \right)
    \]
    """

    def f(self,s):
        # s=s.tolist()[0]
        self.t_span
        t_eval = np.linspace(0, self.t_span, self.integration_steps)
        return solve_ivp(self.system, self.t_span, s, t_eval=t_eval)

    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError
    
    def attractors(self):
        return [[-1,0], [1,0]]

    def which_attracting_region(self, s):
        """defaut -1: attracting region not specified"""
        if np.linalg.norm(s - np.array([1,0]))<0.1:
            return 1
        elif np.linalg.norm(s - np.array([-1,0]))<0.1:
            return 0
        else:
            return False