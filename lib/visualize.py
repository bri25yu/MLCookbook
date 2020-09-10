"""
@author bri25yu
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import LogNorm
from matplotlib import animation
from scipy.optimize import minimize, OptimizeResult

class Visualization:

    @staticmethod
    def visualize2D(dataframe, save_name='visualize2D', show=True):
        sns.heatmap(dataframe, annot=True, cmap=sns.cm.rocket_r)
        plt.tight_layout()
        plt.savefig(save_name)
        if show: plt.show()


class Optimization:
    """
    Author: Suvansh Sanjeev (suvansh@berkeley.edu)
    Course: EECS 127 (UC Berkeley)
    Notes: Parts adapted from http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/

    bri25yu: Slight modification to fit the Function datatype
    """
    
    def __init__(self, methods, functions):
        self.methods = methods
        self.functions = functions
        self.has_set = self.has_set_fn = False
    
    """ For your use """
    
    def set_fn_settings(self, fn_name):
        self.fn_name = fn_name
        self.xmin, self.xmax, self.xstep = self.get_coord_bounds(fn_name)
        self.ymin, self.ymax, self.ystep = self.get_coord_bounds(fn_name)
        self.x, self.y = np.meshgrid(np.arange(self.xmin, self.xmax + self.xstep, self.xstep),
                                     np.arange(self.ymin, self.ymax + self.ystep, self.ystep))
        self.f = self.get_fg(fn_name)
        self.z = self.f((self.x, self.y))[0]
        self.minima_ = self.get_minimum(fn_name)
        self.elev, self.azim = self.get_elev_azim(fn_name)
        self.has_set_fn = True
    
    def set_settings(self, fn_name, method, x0=None, **kwargs):
        if method not in self.methods:
            raise ValueError('Invalid method %s' % method)
        self.set_fn_settings(fn_name)
        self.method = self.methods[method]
        if x0 is None: x0 = self.find(fn_name).initial_values()
        self.x0 = x0
        self.options = kwargs
        path_ = [x0]
        result = minimize(self.f, x0=x0, method=self.method,
                               jac=True, tol=1e-20, callback=self.make_minimize_cb(path_),
                               options=kwargs)
        assert len(result) == 2 and isinstance(result[0], OptimizeResult) and isinstance(result[1], np.ndarray)
        self.res, self.losses = result
        self.path = np.array(path_).T
        self.has_set = True
    
    def get_settings(self):
        return self.fn_name, self.method.__name__, self.x0, self.options
    
    def compare(self, method, start_iter=0, **kwargs):
        res1, losses1 = self.res, self.losses
        curr_settings = self.get_settings()
        self.set_settings(self.fn_name, method, self.x0, **kwargs)
        res2, losses2 = self.res, self.losses
        # plot training curves
        method1 = curr_settings[1]
        method2 = self.method.__name__
        plt.plot(np.arange(len(losses1)-start_iter), losses1[start_iter:], label=method1)
        plt.plot(np.arange(len(losses2)-start_iter), losses2[start_iter:], label=method2)
        plt.title('Training Curve')
        plt.legend()
        plt.show()
        print('[Method {:>10}] Final loss: {:.4f}, Final x: [{:.4f}, {:.4f}]'.format(method1, losses1[-1], res1.x[0], res1.x[1]))
        print('[Method {:>10}] Final loss: {:.4f}, Final x: [{:.4f}, {:.4f}]'.format(method2, losses2[-1], res2.x[0], res2.x[1]))
        self.set_settings(*curr_settings[:-1], **curr_settings[-1])
        
    def plot2d(self):
        self.check_set_fn()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.x, self.y, self.z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.plot(*self.minima_, 'r*', markersize=18)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
    
    def plot3d(self):
        self.check_set_fn()
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection='3d', elev=self.elev, azim=self.azim)
        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=1, cstride=1, 
                        edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        ax.plot(*self.minima_, self.f(self.minima_)[0], 'r*', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(self.method.__name__)
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
        
    def path2d(self):
        self.check_set()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.x, self.y, self.z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.quiver(self.path[0,:-1], self.path[1,:-1], self.path[0,1:]-self.path[0,:-1], self.path[1,1:]-self.path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
        ax.plot(*self.minima_, 'r*', markersize=18)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
        
    def path3d(self):
        self.check_set()
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection='3d', elev=self.elev, azim=self.azim)

        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        ax.quiver(self.path[0,:-1], self.path[1,:-1], self.f(self.path[::,:-1])[0], 
                  self.path[0,1:]-self.path[0,:-1], self.path[1,1:]-self.path[1,:-1],
                  self.f(self.path[::,1:])[0]-self.f(self.path[::,:-1])[0], 
                  normalize=True, color='k')
        ax.plot(*self.minima_, self.f(self.minima_)[0], 'r*', markersize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(self.method.__name__)
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        plt.show()
        
    def video2d(self):
        self.check_set()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.contour(self.x, self.y, self.z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
        ax.plot(*self.minima_, 'r*', markersize=18)
        line, = ax.plot([], [], 'b', label=self.method.__name__, lw=2)
        point, = ax.plot([], [], 'bo')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        ax.legend(loc='upper left')
        anim = animation.FuncAnimation(fig, self.get_animate2d(line, point), init_func=self.get_init2d(line, point),
                                       frames=self.path.shape[1], interval=60, 
                                       repeat_delay=5, blit=True)
        filename = filename or '%s_%s_2d.mp4' % (self.fn_name, self.method.__name__)
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        anim.save(filename, writer=animation.writers['imagemagick'](fps=15))
        
    def video3d(self, filename=None):
        self.check_set()
        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection='3d', elev=self.elev, azim=self.azim)
        ax.plot_surface(self.x, self.y, self.z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        ax.plot(*self.minima_, self.f(self.minima_)[0], 'r*', markersize=10)
        line, = ax.plot([], [], [], 'b', label=self.method.__name__, lw=2)
        point, = ax.plot([], [], [], 'bo')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel(self.method.__name__)
        ax.set_xlim((self.xmin, self.xmax))
        ax.set_ylim((self.ymin, self.ymax))
        anim = animation.FuncAnimation(fig, self.get_animate3d(line, point), init_func=self.get_init3d(line, point),
                                       frames=self.path.shape[1], interval=60, 
                                       repeat_delay=5, blit=True)
        filename = filename or '%s_%s_3d.mp4' % (self.fn_name, self.method.__name__)
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        anim.save(filename, writer=animation.writers['imagemagick'](fps=15))
        
    def get_xs_losses(self):
        self.check_set()
        return self.path.T, self.losses
    
    def get_min_errs(self):
        """ Returns the best x differences and function differences over the run. """
        x_err = np.linalg.norm(self.path - self.minima_, axis=0).min()
        loss_err = (self.losses - self.f(self.minima_)[0]).min()
        return x_err, loss_err
    
    def func_val(self, x):
        return self.f(x)[0]
    
    def grad_val(self, x):
        return self.f(x)[1]

    """ Under the hood """
    
    def check_set_fn(self):
        assert self.has_set_fn, "Need to call `set_fn_settings` first."
    
    def check_set(self):
        assert self.has_set, "Need to call `set_settings` first."
        
    def find(self, fn):
        try:
            return self.functions[fn]
        except:
            raise ValueError('Invalid function %s' % fn)

    def get_fg(self, fn_name):
        func = self.find(fn_name)
        return lambda x: (func.value(x[0], x[1]), func.grad(x[0], x[1]))
    
    def get_coord_bounds(self, fn):
        return self.find(fn).coord_bounds()
    
    def get_minimum(self, fn):
        return self.find(fn).minimum()

    def get_elev_azim(self, fn):
        return self.find(fn).elev_azim()
        
    def make_minimize_cb(self, path=[]):
        return lambda xk: path.append(np.copy(xk))
    
    def get_init2d(self, line, point):
        def init2d():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        return init2d
    
    def get_animate2d(self, line, point):
        def animate2d(i):
            line.set_data(*self.path[::,:i])
            point.set_data(*self.path[::,i-1:i])
            return line, point
        return animate2d
        
    def get_init3d(self, line, point):
        def init3d():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        return init3d

    def get_animate3d(self, line, point):
        def animate3d(i):
            line.set_data(self.path[0,:i], self.path[1,:i])
            line.set_3d_properties(self.f(self.path[::,:i])[0])
            point.set_data(self.path[0,i-1:i], self.path[1,i-1:i])
            point.set_3d_properties(self.f(self.path[::,i-1:i])[0])
            return line, point
        return animate3d


class AnimateXYZ:
    DEFAULT_INTERVAL = 50  # in ms
    COLORS = ['red', 'green', 'blue']

    def __init__(self, lines_data: list, arrows_data: list=None):
        """
        Parameters
        ----------
        lines_data: list
            List of n lines, where each line is an np.array of shape (3, steps).
        arrows_data: list
            List of n arrow sets. Each arrow set contains 3 directions. Each arrow direction is
            an np.array of shape (3, steps).

        """
        self.lines_data, self.arrows_data = lines_data, arrows_data
        self.num_frames = self.lines_data[0].shape[1]
        self.fig, self.ax = self.create_3d()
        self.plot_lines()
        self.plot_arrows()
        self.create_animation()

    @staticmethod
    def update_lines_and_arrows(t, lines_data, lines, arrows_data, arrows):
        objects_to_plot = AnimateXYZ.update_lines(t, lines_data, lines)
        if arrows_data:
            objects_to_plot.extend(AnimateXYZ.update_arrows(t, lines_data, arrows_data, arrows))
        return objects_to_plot

    @staticmethod
    def update_lines(t, lines_data, lines):
        for line_data, line in zip(lines_data, lines):
            line.set_data(line_data[0:2, :t])
            line.set_3d_properties(line_data[2, :t])
        return lines

    @staticmethod
    def arrows_data_to_segments(X, Y, Z, u, v, w):
        """
        From https://stackoverflow.com/questions/48911643/set-uvc-equivilent-for-a-3d-quiver-plot-in-matplotlib
        """
        segments = (X, Y, Z, X + u, Y + v, Z + w)
        segments = np.array(segments).reshape(6,-1)
        return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

    @staticmethod
    def update_arrows(t, lines_data, arrows_data, arrows):
        for line_data, triple_arrow_data, triple_arrow in zip(lines_data, arrows_data, arrows):
            x, y, z = AnimateXYZ.get_current(line_data, t)
            for arrow, arrow_data in zip(triple_arrow, triple_arrow_data):
                u, v, w = AnimateXYZ.get_current(arrow_data, t)
                segments = AnimateXYZ.arrows_data_to_segments(x, y, z, u, v, w)
                arrow.set_segments(segments)
        return arrows

    @staticmethod
    def get_current(data, t):
        x, y, z = data[0, t:t + 1], data[1, t:t + 1], data[2, t:t + 1]
        return x, y, z

    @staticmethod
    def create_3d():
        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        return fig, ax

    def set_ax(self, x_interval, y_interval, z_interval):
        self.ax.set_xlim3d(x_interval)
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d(y_interval)
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d(z_interval)
        self.ax.set_zlabel('Z')

    def plot_lines(self):
        self.lines = []
        for line_data in self.lines_data:
            x, y, z = self.get_current(line_data, 0)
            self.lines.append(self.ax.plot(x, y, z)[0])

    def plot_arrows(self):
        if self.arrows_data:
            self.arrows = []
            for line_data, triple_arrow_data in zip(self.lines_data, self.arrows_data):
                triple_arrows = []
                for arrow_data, color in zip(triple_arrow_data, self.COLORS):
                    x, y, z = self.get_current(line_data, 0)
                    u, v, w = self.get_current(arrow_data, 0)
                    triple_arrows.append(
                        self.ax.quiver(x, y, z, u, v, w, color=color, length=1, normalize=True))
                self.arrows.append(triple_arrows)

    def create_animation(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_lines_and_arrows,
            self.num_frames,
            fargs=(self.lines_data, self.lines, self.arrows_data, self.arrows),
            interval=self.DEFAULT_INTERVAL,
            blit=False
        )

    def show(self):
        plt.show()

    def save(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        self.ani.save('output.gif', writer=writer)
