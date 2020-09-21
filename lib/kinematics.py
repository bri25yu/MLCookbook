"""
@author bri25yu

A small kinematics library used by animate.py.
"""

import numpy as np

from functools import reduce


def check_shape(func, *shape_args, **shape_kwargs):
    def wrapper(*args, **kwargs):
        for arg, shape_arg in zip(args, shape_args):
            if isinstance(arg, np.ndarray):
                assert arg.shape == shape_arg, f"Argument shape is {arg.shape}, but should be {shape_arg}!"
        
        for name in shape_kwargs:
            if name not in kwargs and not isinstance(kwargs[name], np.ndarray): continue
            arg, shape_arg = kwargs[name], shape_kwargs[name]
            assert arg.shape == shape_arg, f"Argument shape is {arg.shape}, but should be {shape_arg}!"

        func(*args, **kwargs)

    return wrapper


class E:
    X = np.array([[1], [0], [0]], dtype=np.float)
    Y = np.array([[0], [1], [0]], dtype=np.float)
    Z = np.array([[0], [0], [1]], dtype=np.float)

    @staticmethod
    def VECTORS():
        return [E.X, E.Y, E.Z]


class Rot:
    @staticmethod
    def X(theta):
        """
        Returns the 3D rotation matrix X(theta).
        """
        return np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]], dtype=np.float)

    @staticmethod
    def Y(theta):
        """
        Returns the 3D rotation matrix Y(theta).
        """
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float)

    @staticmethod
    def Z(theta):
        """
        Returns the 3D rotation matrix X(theta).
        """
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]], dtype=np.float)


class RigidMotion:
    SE3_BASE = np.array([0, 0, 0, 1], dtype=np.float)

    @staticmethod
    def get_pos(g: np.ndarray) -> np.ndarray:
        """
        Gets the position component of a g matrix.

        Parameters
        ----------
        g: np.ndarray
            A (4, 4)-shaped matrix.
        
        Returns
        -------
        position: np.ndarray
            A (3,)-shaped vector representing the position specified by g.

        """
        return g[0:3, 3]

    @staticmethod
    def get_rotation(g: np.ndarray) -> np.ndarray:
        """
        Gets the rotation component of a g matrix.

        Parameters
        ----------
        g: np.ndarray
            A (4, 4)-shaped matrix.
        
        Returns
        -------
        rotation: np.ndarray
            A (3,3)-shaped matrix representing the rotation specified by g.

        """
        return g[0:3, 0:3]

    @staticmethod
    def skew_3d(omega: np.ndarray) -> np.ndarray:
        """
        Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.
        
        Parameters
        ----------
        omega: np.ndarray
            A (3,)-shaped vector that represents the current rotation.
        
        Returns
        -------
        omega_hat: np.ndarray
            A (3,3)-shaped matrix that represents the corresponding skew symmetric matrix.

        """
        omega_hat = np.zeros((3, 3))

        omega_hat[0, 1] = -omega[2]
        omega_hat[1, 0] = omega[2]

        omega_hat[2, 0] = -omega[1]
        omega_hat[0, 2] = omega[1]

        omega_hat[1, 2] = -omega[0]
        omega_hat[2, 1] = omega[0]

        return omega_hat

    @staticmethod
    def rotation_3d(omega: np.ndarray, theta: float) -> np.ndarray:
        """
        Computes a 3D rotation matrix given a rotation axis and angle of rotation.
        
        Parameters
        ----------
        omega: np.ndarray
            A (3,)-shaped vector that represents the axis of rotation.
        theta: float
            The angle of rotation.
        
        Returns
        -------
        rot: np.ndarray
            A (3,3)-shaped matrix representing the resulting rotation.

        """
        I = np.eye(3)
        omega_hat = RigidMotion.skew_3d(omega)
        mag_omega = np.sqrt(omega.dot(omega))

        first_order = (np.sin(theta * mag_omega) / mag_omega) * omega_hat
        second_order = ((1 - np.cos(theta * mag_omega)) / (mag_omega ** 2)) * np.linalg.matrix_power(omega_hat, 2)
        return I + first_order + second_order

    @staticmethod
    def hat_3d(xi: np.ndarray) -> np.ndarray:
        """
        Converts a 3D twist to its corresponding 4x4 matrix representation.
        
        Parameters
        ----------
        xi: np.ndarray
            A (6,)-shaped vector representing the 3D twist.
        
        Returns
        -------
        xi_hat: np.ndarray
            A (4,4)-shaped matrix representing the resulting twist matrix.

        """
        v, w = xi[:3], xi[3:]
        xi_hat = np.zeros((4, 4))
        xi_hat[:3, :3] = RigidMotion.skew_3d(w)
        xi_hat[:3, 3] = v

        return xi_hat

    @staticmethod
    def homog_3d(xi: np.ndarray, theta: float) -> np.ndarray:
        """
        Computes a 4x4 homogeneous transformation matrix given a 3D twist and a 
        joint displacement.
        
        Parameters
        ----------
        xi: np.ndarray
            An (6,)-shaped vector representing the 3D twist.
        theta: float
            The joint displacement.

        Returns
        -------
        g: np.ndarray
            A (4,4)-shaped matrix representing the resulting homogeneous transformation.

        """
        v, w, I = xi[:3], xi[3:], np.eye(3)
        if np.allclose(w, 0):  # Pure translation
            R = I
            p = v * theta
        else:  # Translation and rotation
            mag_w = np.sqrt(w.dot(w))
            w_hat = RigidMotion.skew_3d(w)
            R = RigidMotion.rotation_3d(w, theta)
            p = (1 / (mag_w ** 2)) * ((I - R).dot(w_hat.dot(v)) + np.outer(w, w).dot(v) * theta)
        g = np.eye(4)
        g[:3, :3] = R
        g[:3, 3] = p
        return g

    @staticmethod
    def prod_exp(xis: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """
        Computes the product of exponentials for a kinematic chain, given 
        the twists and displacements for each joint.
        
        Parameters
        ----------
        xis: np.ndarray
            A (6, N)-shaped matrix that represents the twists for each joint.
        thetas: np.ndarray
            A (N,)-shaped vector representing the displacement in rad of each joint.
        
        Returns
        -------
        g: np.ndarray
            A (4,4)-shaped matrix representing the resulting homogeneous transformation.
        """
        return reduce(lambda xi_hat1, xi_hat2: xi_hat1.dot(xi_hat2), map(lambda p: RigidMotion.homog_3d(*p), zip(xis.T, thetas)))
