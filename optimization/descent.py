"""
@author bri25yu
Credit to 127 staff for skeleton code!
"""

import numpy as np
from scipy.optimize import OptimizeResult

import warnings
warnings.simplefilter('ignore')

from lib.algorithms import HyperparameterTuning
from lib.functions import FUNCTIONS
from lib.visualize import Optimization
from optimization.settings import OUTPUT_DIR

class Descent:
    NAME = 'Descent'
    LR = lambda t: 0.1
    
    def __call__(self, func, x, lr, num_iters, jac, tol, callback, *args, **kwargs):
        return self.optimize(func, x, lr, num_iters, jac, tol, callback, *args, **kwargs)

    def initialize(self, func, x, lr, num_iters, jac, tol, callback, *args, **kwargs):
        self.__name__ = self.NAME # here to satisfy Answer.video3d
        self.lr, self.jac = lr, jac

    def update(self, x, itr):
        return x, x

    def optimize(self, func, x, lr, num_iters, jac, tol, callback, *args, **kwargs):
        self.initialize(func, x, lr, num_iters, jac, tol, callback, *args, **kwargs)
        losses = []
        losses.append(func(x))
        converged = False
        for itr in range(num_iters):
            x, x_old = self.update(x, itr)
            
            callback(x)
            losses.append(func(x))
            if np.linalg.norm(x - x_old) < tol:
                converged = True
                break
            
        opt_obj = OptimizeResult(x=x, success=converged, nit=itr+1)
        return opt_obj, np.array(losses)

    def get_tuning_frame(self, ht, func_name, num_iters, output, **kwargs):
        fn_hyp = ht.Hyperparameter('name', values=[func_name])
        method_hyp = ht.Hyperparameter('method', values=[self.NAME])
        num_iters_hyp = ht.Hyperparameter('num_iters', values=[num_iters])
        output_hyp = ht.Hyperparameter('output', values=[output])

        lr_values = kwargs.get('lr_values', None)
        if lr_values is not None:
            lr_hyp = ht.Hyperparameter('lr', values=lr_values)
        else:
            lr_base = kwargs.get('lr_base', 0.0)
            lr_delta = kwargs.get('lr_delta', 10**6)
            low, high = lr_base - lr_delta, lr_base + lr_delta
            lr_hyp = ht.Hyperparameter('lr', map_fn=lambda v: lambda itr: 10**v, start=low, end=high, num_values=10**2)

        frame = ht.HyperparameterFrame([fn_hyp, lr_hyp, method_hyp, num_iters_hyp, output_hyp])
        return frame

    def get_check_function_params(self):
        ans = Optimization(DESCENT_METHODS, FUNCTIONS)
        self.ans = ans
        def check_function_params(**kwargs):
            output = kwargs.pop('output') if 'output' in kwargs else True
            name = kwargs.pop('name')
            params = kwargs
            
            ans.set_settings(fn_name=name, **params)
            x_diff, loss_diff = ans.get_min_errs()

            min_x_diff, min_loss_diff = FUNCTIONS[name].thresholds()
            x_score, loss_score = x_diff - min_x_diff, loss_diff - min_loss_diff

            x_passed = 'Passed (%s)' % x_diff if x_score < 0 else x_diff
            loss_passed = 'Passed (%s)' % loss_diff if loss_score < 0 else loss_diff

            if output: print(name, params['method'], 'x:', x_passed, 'loss:', loss_passed, 'nit:', ans.res.nit)

            return x_diff, params
        return check_function_params

    def tune(self, func_name, num_iters=100, output=False, **kwargs):
        self.ht = HyperparameterTuning(None, None)
        frame = self.get_tuning_frame(self.ht, func_name, num_iters, output, **kwargs)
        check_function_params = self.get_check_function_params()
        optimized = self.ht.grid_search(frame, check_function_params)
        print(optimized, optimized[1]['lr'](100))
        return optimized

    def visualize_tuning(self):
        self.ht.visualize2D('lr', 'method', path=OUTPUT_DIR, show=False)

class GradientDescent(Descent):
    NAME = 'gd'
    LR = lambda t: 0.25
    
    def update(self, x, itr):
        x_old = x
        x = x_old - self.lr(itr) * self.jac(x_old)
        return x, x_old

class MomentumGradientDescent(Descent):
    NAME = 'mgd'
    ALPHA = 0.9
    LR = lambda t: 0.06
    
    def initialize(self, func, x, lr, num_iters, jac, tol, callback, *args, **kwargs):
        super().initialize(func, x, lr, num_iters, jac, tol, callback, *args, **kwargs)
        self.alpha = kwargs.get('alpha', MomentumGradientDescent.ALPHA)
        self.delta = np.zeros(jac(x).shape)

    def update(self, x, itr):
        x_old = x
        self.delta = -self.alpha * self.delta + self.lr(itr) * self.jac(x_old)
        x = x_old - self.delta
        return x, x_old

    def get_tuning_frame(self, ht, func_name, num_iters, output, **kwargs):
        frame = super().get_tuning_frame(ht, func_name, num_iters, output, **kwargs)

        alpha_values = kwargs.get('alpha_values', None)
        if alpha_values is not None:
            alpha_hyp = ht.Hyperparameter('alpha', values=alpha_values)
        else:
            alpha_base = kwargs.get('alpha_base', 0.0)
            alpha_delta = kwargs.get('alpha_delta', 10**6)
            low, high = alpha_base - alpha_delta, alpha_base + alpha_delta
            alpha_hyp = ht.Hyperparameter('alpha', map_fn=lambda v: lambda itr: 10**v, start=low, end=high, num_values=10**2)
        frame.append(alpha_hyp)
        return frame

    def visualize_tuning(self):
        self.ht.visualize2D('lr', 'alpha', path=OUTPUT_DIR, show=False)

class NesterovAcceleratedGradientDescent(MomentumGradientDescent):
    NAME = 'nad'
    LR = lambda t: 0.4

    def update(self, x, itr):
        x_old = x
        
        self.delta = -self.alpha * self.delta + self.lr(itr) * self.jac(x + self.alpha * self.delta)
        x = x_old - self.delta
        return x, x_old

class AdaptiveGradientDescent(Descent):
    NAME = 'adagrad'
    EPS = 1e-8
    LR = lambda t: 2
    
    def initialize(self, func, x, lr, num_iters, jac, tol, callback, *args, **kwargs):
        super().initialize(func, x, lr, num_iters, jac, tol, callback, *args, **kwargs)
        self.eps = kwargs.get('eps', AdaptiveGradientDescent.EPS)
        self.delta = np.zeros(jac(x).shape)

    def update(self, x, itr):
        x_old = x
        
        dx = self.jac(x)
        self.delta = self.delta + dx ** 2
        x = x_old - self.lr(itr) * dx / (np.sqrt(self.delta) + self.eps)
        return x, x_old

class Adadelta(Descent):
    NAME = 'adadelta'
    EPS = 1e-5
    ALPHA = 0.9
    
    def initialize(self, func, x, lr, num_iters, jac, tol, callback, *args, **kwargs):
        super().initialize(func, x, lr, num_iters, jac, tol, callback, *args, **kwargs)
        self.alpha = kwargs.get('alpha', Adadelta.ALPHA)
        self.eps = kwargs.get('eps', Adadelta.EPS)
        self.gradient = np.zeros(jac(x).shape)
        # self.update = np.zeros()

    def rms(self, val):
        return np.sqrt(val + self.eps)

    def update(self, x, itr):
        x_old = x
        
        dx = self.jac(x)
        self.delta = self.delta + dx ** 2
        x = x_old - self.lr(itr) * dx / (np.sqrt(self.delta) + self.eps)
        return x, x_old

DESCENT_METHODS = [GradientDescent, MomentumGradientDescent, NesterovAcceleratedGradientDescent, AdaptiveGradientDescent, Adadelta]
DESCENT_METHODS = {method.NAME : method() for method in DESCENT_METHODS}
