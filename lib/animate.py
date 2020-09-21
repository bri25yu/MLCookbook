"""
@author bri25yu

A small animation library for animating 3 space!
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from .kinematics import E, RigidMotion


class AnimateXYZ:
    DEFAULT_INTERVAL = 50  # in ms
    COLORS = ['red', 'green', 'blue']
    AXES_SCALING_FACTOR = 1.2
    DEFAULT_FONT_SIZE = 12

    def __init__(self,
        lines_data: list,
        arrows_data: list=None,
        segments_data: list=None,
        line_names: list=None,
        update_fn_wrapper=None,
        frames=None,
        repeat: bool=True):
        """
        Parameters
        ----------
        lines_data: list
            List of n lines, where each line is an np.array of shape (3, steps).
        arrows_data: list
            List of n arrow sets. Each arrow set contains 3 directions. Each arrow direction is
            an np.array of shape (3, steps).
        segments_data: list
            List of lines to connect, in the form [[l1, l2], ...].
        line_names: list
            List of names corresponding to the lines in lines_data.
        update_fn_wrapper: fn
            update_fn_wrapper used in AnimateXYZ.create_animation
        frames
            Frames specification.
        repeat: bool
            Whether or not to repeat the animation.

        """
        self.lines_data = lines_data
        self.arrows_data = arrows_data if arrows_data is not None else []
        self.segments_data = segments_data if segments_data is not None else []
        self.line_names = line_names if line_names is not None else []

        self.set_update_fn_wrapper(update_fn_wrapper)
        self.set_frames(frames)

        self.fig, self.ax = self.create_3d()
        self.plot_lines()
        self.plot_arrows()
        self.plot_segments()
        self.annotate_lines()
        scaled = np.max(lines_data) * AnimateXYZ.AXES_SCALING_FACTOR
        AXES = [-scaled, scaled]
        self.set_ax(AXES, AXES, AXES)
        self.create_animation()

    @staticmethod
    def update(
        t,
        lines_data,
        lines,
        arrows_data,
        arrows,
        segments_data,
        segments,
        line_names,
        annotations,
        prev_t=0):
        objects_to_plot = AnimateXYZ.update_lines(t, lines_data, lines, prev_t)
        objects_to_plot.extend(AnimateXYZ.update_arrows(t, lines_data, arrows_data, arrows))
        objects_to_plot.extend(AnimateXYZ.update_segments(t, lines_data, segments_data, segments))
        objects_to_plot.extend(AnimateXYZ.update_annotations(t, lines_data, line_names, annotations))
        return objects_to_plot

    @staticmethod
    def update_lines(t, lines_data, lines, prev_t=0):
        for line_data, line in zip(lines_data, lines):
            line.set_data(line_data[0:2, prev_t:t])
            line.set_3d_properties(line_data[2, prev_t:t])
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
    def update_segments(t, lines_data, segments_data, segments):
        for (i1, i2), segment in zip(segments_data, segments):
            l1_data, l2_data = lines_data[i1], lines_data[i2]
            x1, y1, z1 = AnimateXYZ.get_current(l1_data, t)
            x2, y2, z2 = AnimateXYZ.get_current(l2_data, t)

            x = np.array(np.ravel([x1, x2]))
            y = np.array(np.ravel([y1, y2]))
            z = np.array(np.ravel([z1, z2]))

            segment.set_data(x, y)
            segment.set_3d_properties(z)
        return segments

    @staticmethod
    def update_annotations(t, lines_data, line_names, annotations):
        for lines_datum, _, annotation in zip(lines_data, line_names, annotations):
            x, y, z = np.ravel(AnimateXYZ.get_current(lines_datum, t))
            annotation.set_position((x, y))
            annotation.set_3d_properties(z)
        return annotations

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
        self.arrows = []
        for line_data, triple_arrow_data in zip(self.lines_data, self.arrows_data):
            triple_arrows = []
            for arrow_data, color in zip(triple_arrow_data, self.COLORS):
                x, y, z = self.get_current(line_data, 0)
                u, v, w = self.get_current(arrow_data, 0)
                triple_arrows.append(
                    self.ax.quiver(x, y, z, u, v, w, color=color, length=1, normalize=True))
            self.arrows.append(triple_arrows)

    def plot_segments(self):
        self.segments = []
        for i1, i2 in self.segments_data:
            l1_data, l2_data = self.lines_data[i1], self.lines_data[i2]
            x1, y1, z1 = self.get_current(l1_data, 0)
            x2, y2, z2 = self.get_current(l2_data, 0)
            
            x = np.array(np.ravel([x1, x2]))
            y = np.array(np.ravel([y1, y2]))
            z = np.array(np.ravel([z1, z2]))

            self.segments.append(self.ax.plot(x, y, z)[0])

    def annotate_lines(self):
        self.annotations = []
        for line_data, line_name in zip(self.lines_data, self.line_names):
            x, y, z = np.ravel(self.get_current(line_data, 0))
            self.annotations.append(self.ax.text(x, y, z, line_name, fontsize=AnimateXYZ.DEFAULT_FONT_SIZE))

    @staticmethod
    def default_update_fn_wrapper(fn):
        return fn

    def create_animation(self):
        update_fn = self.update_fn_wrapper(self.update)

        self.ani = animation.FuncAnimation(
            self.fig,
            update_fn,
            frames=self.frames,
            fargs=(self.lines_data,
                self.lines,
                self.arrows_data,
                self.arrows,
                self.segments_data,
                self.segments,
                self.line_names,
                self.annotations,),
            interval=self.DEFAULT_INTERVAL,
            blit=False,
        )

    def set_frames(self, frames=None):
        if frames is None:
            frames = self.lines_data.shape[-1] - 1
        elif frames == "inf":
            frames = None
        self.frames = frames

    def set_update_fn_wrapper(self, update_fn_wrapper=None):
        if update_fn_wrapper is None:
            update_fn_wrapper = self.default_update_fn_wrapper
        self.update_fn_wrapper = update_fn_wrapper

    def show(self):
        plt.show()

    def save(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        self.ani.save('output.gif', writer=writer)


class AnimateForwardKinematics:
    DEFAULT_NUM_THETA = 180

    def __init__(self, foward_kinematics_fn, num_joints: int, joint_names: list=None):
        """
        Parameters
        ----------
        forward_kinematics_fn: fn
            Takes in `num_joints` number of theta values and returns (g_{st}(theta), xi_array).
        num_joints: int
            Number of joints to animate.

        """
        self.fn, self.num_joints = foward_kinematics_fn, num_joints
        self.joint_names = joint_names

    def animate_isolate(self, num: int):
        """
        Animates only a single joint from theta = [0, pi].

        Parameters
        ----------
        num: int
            The number corresponding to the joint to animate from [0, num_joints-1].

        """
        thetas = self.get_single_joint(num, 0, 2 * np.pi)
        self.animate(thetas)

    def animate(self, thetas: np.ndarray):
        """
        Animates using self.fn for all of the input thetas.

        Parameters
        ----------
        thetas: np.ndarray
            A (num, self.num_joints)-shaped np array.

        """
        lines_data, arrows_data, segments_data, joint_names = self.get_animation_data(thetas)
        self.animation = AnimateXYZ(lines_data, arrows_data, segments_data, joint_names)
        self.animation.show()

    def animate_interactive(self, save=True):
        thetas = np.zeros((1, self.num_joints))
        lines_data, arrows_data, segments_data, joint_names = self.get_animation_data(thetas)
        self.animation = AnimateXYZ(
            lines_data,
            arrows_data,
            segments_data,
            joint_names,
            lambda fn: self.update_fn_wrapper(fn),
            frames="inf",)

        print("Press 'q' + Enter to quit.")
        self.animation.show()

        if save:
            self.animation.set_frames()
            self.animation.set_update_fn_wrapper()
            self.animation.create_animation()
            self.save()

    def save(self):
        self.animation.save()

    def get_animation_data(self, thetas):
        lines_data = np.empty((self.num_joints + 1, 3, thetas.shape[0]))
        arrows_data = np.empty((self.num_joints + 1, 3, 3, thetas.shape[0]))
        segments_data = AnimateForwardKinematics.get_adjacent_pairs(self.num_joints)

        for t, theta in enumerate(thetas):
            gs = self.theta_to_gs(theta)

            for j, g in enumerate(gs):
                line_datum, arrow_datum = AnimateForwardKinematics.g_to_draw_data(g)
                lines_data[j, :, t], arrows_data[j, :, :, t] = line_datum, arrow_datum

        return lines_data, arrows_data, segments_data, self.joint_names

    def update_fn_wrapper(self, fn):
        thetas = np.zeros(self.num_joints)
        prev_t, to_quit = 0, False
        def update_fn(
            t,
            lines_data,
            lines,
            arrows_data,
            arrows,
            segments_data,
            segments,
            line_names,
            annotations
        ):
            nonlocal prev_t, to_quit
            if to_quit: return

            if t == self.animation.lines_data.shape[-1]:
                joint_num, joint_name, new_to_quit = AnimateForwardKinematics.get_joint_name(self.num_joints, self.joint_names)
                to_quit = to_quit or new_to_quit
                if to_quit:
                    plt.close()
                    return

                additional_theta, new_to_quit = AnimateForwardKinematics.get_additional_theta(joint_name)
                to_quit = to_quit or new_to_quit
                if to_quit:
                    plt.close()
                    return
                
                previous_theta = thetas[joint_num]
                thetas[joint_num] = previous_theta + additional_theta
                new_thetas = self.get_single_joint(joint_num, previous_theta, thetas[joint_num])
                for i, theta in enumerate(thetas):
                    if i == joint_num: continue
                    new_thetas[:, i] = theta

                new_lines_data, new_arrows_data, _, _ = self.get_animation_data(new_thetas)

                self.animation.lines_data = AnimateForwardKinematics.combine(self.animation.lines_data, new_lines_data)
                self.animation.arrows_data = AnimateForwardKinematics.combine(self.animation.arrows_data, new_arrows_data)

                prev_t = t

            fn(
                t,
                self.animation.lines_data,
                lines,
                self.animation.arrows_data,
                arrows,
                segments_data,
                segments,
                line_names,
                annotations,
                prev_t=prev_t,
            )

        return update_fn

    @staticmethod
    def get_joint_name(num_joints: int, joint_names: list) -> tuple:
        """
        Gets a user input joint name.

        Parameters
        ----------
        num_joints: int
            The number of joints.
        joint_names: list
            The list of joint names to convert the input joint number into.

        Returns
        -------
        joint_num: int
            Number of joint selected by the user. None if the user wants to quit.
        joint_name: str
            Name of the joint selected by the user. None if the user wants to quit.
        to_quit: bool
            A bool representing whether or not to quit the interactive session.

        """
        joint_num, joint_name = None, None
        to_quit = False
        while not joint_name:
            try:
                user_input = input(f"Please enter a joint number [0, {num_joints - 1}]: ")
                joint_num = int(user_input)
                joint_name = joint_names[joint_num] if joint_names else f"joint {joint_num}"
            except:
                if user_input == 'q':
                    to_quit = True
                    break
                print(f"{joint_num} is not a valid joint number.")

        return joint_num, joint_name, to_quit

    @staticmethod
    def get_additional_theta(joint_name: str) -> tuple:
        """
        Gets a user input for additional angle to traverse.

        Parameters
        ----------
        joint_name: str
            Name of joint.

        Returns
        -------
        additional_theta: float
            The additional angle amount to traverse. None if the user wants to quit.
        to_quit: bool
            A bool representing whether or not to quit the interactive session.

        """
        additional_theta = None
        to_quit = False
        while not additional_theta:
            try:
                user_input = input(f"Please enter an angle in degrees to add to the current position of {joint_name}: ")
                additional_theta = float(user_input)
            except:
                if user_input == 'q':
                    to_quit = True
                    break
                print(f"{additional_theta} is not a valid angle.")

        additional_theta = additional_theta * np.pi / 180

        return additional_theta, to_quit

    @staticmethod
    def combine(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """
        Non-destructively combines arr1 and arr2 along their last axis. arr1 and arr2 are both 3D.

        Parameters
        ----------
        arr1: np.ndarray
            The 3D array to copy first.
        arr2: np.ndarray
            The 3D array to copy second.

        """
        return np.concatenate((arr1, arr2), axis=-1)

    def get_single_joint(self, joint_num: int, theta_0: float, theta_1: float) -> np.ndarray:
        """
        Returns an array of thetas where all values are 0 except for the column specified by
        joint_num where the values are interpolated from [theta_0, theta_1].

        Parameters
        ----------
        joint_num: int
            Number corresponding to the joint to interpolate thetas for.
        theta_0: float
            The current theta.
        theta_1: float
            The next theta to interpolate to.

        Returns
        -------
        thetas: np.ndarray
            An matrix of theta values.

        """
        steps = max(int(self.DEFAULT_NUM_THETA * (np.abs(theta_0 - theta_1) / (2 * np.pi))), 1)
        theta = np.linspace(theta_0, theta_1, num=steps, endpoint=True)
        thetas = np.zeros((theta.shape[0], self.num_joints))
        thetas[:, joint_num] = theta
        return thetas

    def theta_to_gs(self, theta):
        g, xis = self.fn(theta)

        gs, current = [], np.eye(4)
        for xi, t in zip(xis.T, theta):
            current = current @ AnimateForwardKinematics.xi_to_g(xi, t)
            gs.append(current)
        gs.append(g)

        return gs

    @staticmethod
    def xi_to_g(xi, theta):
        return RigidMotion.homog_3d(xi, theta)

    @staticmethod
    def g_to_draw_data(g):
        line_datum = RigidMotion.get_pos(g)

        arrow_datum = np.empty((3, 3))
        current_rotation = RigidMotion.get_rotation(g)
        for i, e in enumerate(E.VECTORS()):
            arrow_datum[i, :] = np.ravel(current_rotation @ e)

        return line_datum, arrow_datum

    @staticmethod
    def get_adjacent_pairs(n):
        return [[i, i + 1] for i in range(n)]


if __name__ == "__main__":
    def gen_rand_line(length):
        """
        Generates a random 3D line.
        """
        lineData = np.empty((3, length))
        lineData[:, 0] = np.random.rand(3)
        for index in range(1, length):
            step = ((np.random.rand(3) - 0.5) * 0.1)
            lineData[:, index] = lineData[:, index - 1] + step
        return lineData

    np.random.seed(1234)

    data = [gen_rand_line(25) for index in range(50)]
    axes = [0.0, 1.0]

    animation = AnimateXYZ(data)
    animation.set_ax(axes, axes, axes)
    animation.ax.set_title('3D Test')
    animation.show()
