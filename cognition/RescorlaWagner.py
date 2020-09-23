"""
@author bri25yu
"""

import numpy as np

import matplotlib.pyplot as plt

from copy import deepcopy


OUTPUT_DIR = "output/"


def main():
    rw = RescorlaWagner(initialize_default=True)
    rw.demonstration()


class RescorlaWagner:
    """
    A single Rescorla-Wagner experiment.
    """
    DEFAULT_NUM_TRIALS = 100
    US_SALIENCE = 1

    class State:
        def __init__(self,
            saliences: np.ndarray,
            uncertainties: np.ndarray,
            association_strengths: np.ndarray):
            self.saliences = saliences
            self.uncertainties = uncertainties
            self.association_strengths = association_strengths

    class Reinforcement:
        def __init__(self, stimuli: np.ndarray):
            """
            stimuli: np.ndarray
                A (n,)-shaped vector of bools representing stimuli present.
            """
            self.stimuli = stimuli

    def __init__(self,
        initial_saliences: np.ndarray=None,
        initial_uncertainties: np.ndarray=None,
        initial_association_strengths: np.ndarray=None,
        learning_rate: float=None,
        initialize_default: bool=False):
        """
        Parameters
        ----------
        initial_saliences: np.ndarray
            An (n,)-shaped vector of initial salience values. The greater the salience, the more
            effective the stimulus. US have salience value 1.
        initial_uncertainties: np.ndarray
            An (n, n)-shaped matrix of initial uncertainty values. The greater the uncertainty, the
            less effective the stimulus. Every stimulus has uncertainty
            0 with itself.
        initial_association_strengths: np.ndarray
            An (n, n)-shaped matrix of initial association strengths. For example,
            as[item_1][item_2] is the association strength of `item_1` conditioned onto `item_2`.
            Every item has association strength of 1 with itself.
        learning_rate: float
            The learning rate.
        initialize_default=False
            Whether or not to initialize this to its default settings.

        """
        self.initialize(
            initial_saliences,
            initial_uncertainties,
            initial_association_strengths,
            learning_rate,
            initialize_default,)

    def initialize(self,
        initial_saliences: np.ndarray=None,
        initial_uncertainties: np.ndarray=None,
        initial_association_strengths: np.ndarray=None,
        learning_rate: float=None,
        initialize_default: bool=False,):
        if initialize_default:
            default_learning_rate, default_saliences, default_uncertainties, default_association_strengths = self.get_default()
            learning_rate = learning_rate if learning_rate is not None else default_learning_rate
            initial_saliences = initial_saliences if initial_saliences is not None else default_saliences
            initial_uncertainties = initial_uncertainties if initial_uncertainties is not None else default_uncertainties
            initial_association_strengths = initial_association_strengths if initial_association_strengths is not None else default_association_strengths

        assert learning_rate is not None and initial_saliences is not None and initial_uncertainties is not None\
            and initial_association_strengths is not None and learning_rate is not None, "Input args must not be empty!"

        assert initial_saliences.shape[0] == initial_uncertainties.shape[0] == initial_uncertainties.shape[1]\
            == initial_association_strengths.shape[0] == initial_association_strengths.shape[1],\
            "saliences, uncertainties, and association strengths must have the same dimensions!"

        self.learning_rate = learning_rate

        self.history = [self.State(initial_saliences, initial_uncertainties, initial_association_strengths)]

    def get_default(self):
        learning_rate = 0.1

        initial_saliences = np.array([0.5, self.US_SALIENCE])
        initial_uncertainties = np.array([
            [0, 0.5],
            [0.5, 0],
        ])
        initial_association_strengths = np.eye(2)

        return learning_rate, initial_saliences, initial_uncertainties, initial_association_strengths

    def run_trials(self, reinforcements: list, stopping_fn=lambda x: None) -> list:
        """
        Parameters
        ----------
        reinforcements: list
            A list of self.Reinforcement objects for every timestep we want to run the trial for.
        stopping_fn: callable
            A function that's called after every time step that indicates whether to stop or not.

        Returns
        -------
        history: list
            A list of self.State objects representing the history of this experiment.

        """
        if not reinforcements: return

        current_state = self.history[-1]
        for reinforcement in reinforcements:
            next_state = self.step(current_state, reinforcement)

            self.history.append(next_state)
            current_state = next_state

            if stopping_fn(current_state): break

        return self.history

    def step(self, state, reinforcement):
        """
        Currently, only allow for multiple values to condition, but only one value to condition on.

        Parameters
        ----------
        state: self.State
            A representation of the current state.
        reinforcement: self.Reinforcement
            A representation of the stimuli currently present.

        Returns
        -------
        next_state: self.State
            A representation of the next state.

        """
        new_saliences = state.saliences.copy()
        new_uncertainties = deepcopy(state.uncertainties)
        new_association_strengths = deepcopy(state.association_strengths)

        r = reinforcement.stimuli
        n = r.shape[0]
        s = state.saliences
        a = state.association_strengths
        u = state.uncertainties

        # This is the start of the update rule

        total_association = np.sum(a, axis=0) - np.diagonal(a)
        p = np.tile(r, (n, 1)).T
        saliences = np.outer(s, s)
        d = r - state.association_strengths

        new_association_strengths += p * self.learning_rate * saliences * (r - total_association)
        new_uncertainties += self.learning_rate * (np.abs(d) - u)

        # This is the end of the update rule

        new_state = self.State(new_saliences, new_uncertainties, new_association_strengths)

        return new_state

    # Plotting ------------------------------------------------------------------------------------

    def plot(self, item1: int, item2: int, save=True, fig=None, ax=None):
        """
        Plots the current history of item1 conditioned on item2.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        association_strengths = self.get_product(self.history, item1, item2)
        ax.plot(association_strengths)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Association strength")
        ax.set_title(f"Association strength over time for item {item1} conditioned on item {item2}")

        if save: fig.savefig(f"{OUTPUT_DIR}as_{item1}_{item2}.png")

        return fig, ax

    # Helpers -------------------------------------------------------------------------------------

    @staticmethod
    def get_product(states, item1, item2) -> np.ndarray:
        """
        Returns the association_strengths * (1 - uncertainties)
        """
        association_strengths = RescorlaWagner.get_association_strengths(states, item1, item2)
        uncertainties = RescorlaWagner.get_uncertainties(states, item1, item2)
        return association_strengths * (1 - uncertainties)

    @staticmethod
    def get_association_strengths(states, item1, item2) -> np.ndarray:
        return np.array([state.association_strengths[item1][item2] for state in states])

    @staticmethod
    def get_uncertainties(states, item1, item2) -> np.ndarray:
        return np.array([state.uncertainties[item1][item2] for state in states])

    # Demonstrations ------------------------------------------------------------------------------

    def demonstration(self):
        """
        A convenience method to run all of the `demonstrate_*` methods.
        """
        self.demonstrate_acquisition()
        self.demonstrate_extinction()
        self.demonstrate_overshadowing()
        self.demonstrate_blocking()
        self.demonstrate_spontaneous_recovery()
        self.demonstrate_history_of_extinction()
        self.demonstrate_second_order()
        self.demonstrate_preexposure()

    def demonstrate_acquisition(self):
        """
        Description: repeated pairings of a conditioned stimulus with an unconditioned stimulus
        increases their association strength.

        Implementation: introduced r, a boolean that represents whether or not the US was present.

        Validation input: association strength of "light" and "food" begin low, then repeated
        pairings of CS "light" and US "food" occur.

        Validation output: upwards trend of association strength between "light" and "food" over time.

        """
        self.initialize(initialize_default=True)

        stimuli = np.array([1, 1])
        reinforcement = self.Reinforcement(stimuli)

        reinforcements = [reinforcement] * self.DEFAULT_NUM_TRIALS

        self.run_trials(reinforcements)

        cs, us = 0, 1
        fig, ax = self.plot(cs, us, save=False)
        ax.set_title(f"Acquisition of item {cs} conditioned on item {us}")
        fig.savefig(f"{OUTPUT_DIR}acquisition.png")

    def demonstrate_extinction(self):
        """
        Description: repeated trials of a conditioned stimulus without the unconditioned
        stimulus after the two have been paired results in decreases in their association strength.

        Implementation: introduced differencing technique of r and the current association strength.

        Validation input: association strength of "light" and "food" begin high, then repeated
        pairings of CS "light" without US "food" occur.

        Validation output: downwards trend of association strength between "light" and "food" over time.

        """
        initial_association_strengths = np.array([[1, 1], [0, 1]], dtype=np.float)
        initial_uncertainties = np.array([[0, 0], [0.5, 0]], dtype=np.float)
        self.initialize(
            initial_association_strengths=initial_association_strengths,
            initial_uncertainties=initial_uncertainties,
            initialize_default=True,
        )

        stimuli = np.array([1, 0])
        reinforcement = self.Reinforcement(stimuli)

        reinforcements = [reinforcement] * self.DEFAULT_NUM_TRIALS

        self.run_trials(reinforcements)

        cs, us = 0, 1
        fig, ax = self.plot(cs, us, save=False)
        ax.set_title(f"Extinction of item {cs} conditioned on item {us}")
        fig.savefig(f"{OUTPUT_DIR}extinction.png")

    def demonstrate_overshadowing(self):
        """
        Description: the growth in association strength are different for different US.

        Implementation: introduced a salience parameter for each US.

        Validation input: association strengths ("light", "food") and ("bell", "food") begin low at
        the same value. "Bell" has a higher salience than "light". Repeated pairings of "light",
        "bell", and "food"

        Validation output: "bell" has a higher association strength with "food" than "light".

        """
        initial_association_strengths = np.eye(3)
        initial_saliences = np.array([0.8, 0.3, self.US_SALIENCE], dtype=np.float)
        initial_uncertainties = np.array([
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ])
        self.initialize(
            initial_association_strengths=initial_association_strengths,
            initial_saliences=initial_saliences,
            initial_uncertainties=initial_uncertainties,
            initialize_default=True,
        )

        stimuli = np.array([1, 1, 1])
        reinforcement = self.Reinforcement(stimuli)

        reinforcements = [reinforcement] * self.DEFAULT_NUM_TRIALS

        self.run_trials(reinforcements)

        cs1, cs2, us = 0, 1, 2
        fig, ax = self.plot(cs1, us, save=False)
        fig, ax = self.plot(cs2, us, save=False, fig=fig, ax=ax)
        ax.set_title(f"Overshadowing of item {cs2} by item {cs1}")
        ax.legend([
            f"Association strength of item {cs1} conditioned on item {us}",
            f"Association strength of item {cs2} conditioned on item {us}",
        ])
        fig.savefig(f"{OUTPUT_DIR}overshadowing.png")

    def demonstrate_blocking(self):
        """
        Description: presenting an non-paired conditioned stimulus while simultaneously presenting
        a paired conditioned stimulus results in the non-paired CS having lower association
        strength.

        Implementation: apply differencing technique for sum over all current association strengths
        related to the particular US.

        Validation input: association strength of "light" and "food" begin high, then repeated
        trials of pairings between "light", "bell", and "food" occur.

        Validation output: association strength of "bell" and "food" increases, but very slowly,
        never reaching maximum strength.

        """
        initial_association_strengths = np.eye(3)
        initial_saliences = np.array([0.2, 0.2, self.US_SALIENCE], dtype=np.float)
        initial_uncertainties = np.array([
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ])
        self.initialize(
            initial_association_strengths=initial_association_strengths,
            initial_saliences=initial_saliences,
            initial_uncertainties=initial_uncertainties,
            initialize_default=True,
        )

        stimuli1 = np.array([1, 0, 1])
        reinforcement1 = self.Reinforcement(stimuli1)
        reinforcements1 = [reinforcement1] * int(self.DEFAULT_NUM_TRIALS * 0.2)

        stimuli2 = np.array([1, 1, 1])
        reinforcement2 = self.Reinforcement(stimuli2)
        reinforcements2 = [reinforcement2] * int(self.DEFAULT_NUM_TRIALS * 0.8)

        reinforcements = reinforcements1 + reinforcements2

        self.run_trials(reinforcements)

        cs1, cs2, us = 0, 1, 2
        fig, ax = self.plot(cs1, us, save=False)
        fig, ax = self.plot(cs2, us, save=False, fig=fig, ax=ax)
        ax.set_title(f"Blocking of item {cs2} by item {cs1}")
        ax.legend([
            f"Association strength of item {cs1} conditioned on item {us}",
            f"Association strength of item {cs2} conditioned on item {us}",
        ])
        fig.savefig(f"{OUTPUT_DIR}blocking.png")

    def demonstrate_spontaneous_recovery(self):
        """
        Description: after extinction, conditioning comes back.

        Implementation: introduce an uncertainty parameter where successive trials without the
        particular CS increases uncertainty, but largely maintains association strength.

        Validation input: association strength of "light" and "food" begin low, then repeated
        pairings occur. After a few trials of pairing, a few trials of "light" presented without
        "food" occur. After that, a few trials of neither "light" nor "food" occur.

        Validation output: Association strength spontaneously increases after extinction has occurred.

        """
        self.initialize(initialize_default=True)

        stimuli1 = np.array([1, 1])
        reinforcement1 = self.Reinforcement(stimuli1)
        reinforcements1 = [reinforcement1] * int(self.DEFAULT_NUM_TRIALS * 0.2)

        stimuli2 = np.array([1, 0])
        reinforcement2 = self.Reinforcement(stimuli2)
        reinforcements2 = [reinforcement2] * int(self.DEFAULT_NUM_TRIALS * 0.3)

        stimuli3 = np.array([0, 0])
        reinforcement3 = self.Reinforcement(stimuli3)
        reinforcements3 = [reinforcement3] * int(self.DEFAULT_NUM_TRIALS * 0.5)

        reinforcements = reinforcements1 + reinforcements2 + reinforcements3

        self.run_trials(reinforcements)

        cs, us = 0, 1
        fig, ax = self.plot(cs, us, save=False)
        ax.set_title(f"Spontaneous recovery of item {cs} conditioned on item {us}")
        fig.savefig(f"{OUTPUT_DIR}spontaneous_recovery.png")

    def demonstrate_history_of_extinction(self):
        """
        Description: extinct stimuli are learned faster.

        Implementation: introduce an uncertainty parameter.

        Validation input: association strength of "light" and "food" begin low, then repeated
        pairings occur. After a few trials of pairing, a few trials of "light" presented without
        "food" occur. After a few trials, repeated trials of "light" and "food" are presented again.

        Validation output: Association strength increasing over the second round of pairing trials,
        but much "faster" than the initial pairing.

        """
        self.initialize(initialize_default=True)

        stimuli1 = np.array([1, 1])
        reinforcement1 = self.Reinforcement(stimuli1)
        reinforcements1 = [reinforcement1] * int(self.DEFAULT_NUM_TRIALS * 0.3)

        stimuli2 = np.array([1, 0])
        reinforcement2 = self.Reinforcement(stimuli2)
        reinforcements2 = [reinforcement2] * int(self.DEFAULT_NUM_TRIALS * 0.4)

        reinforcements = reinforcements1 + reinforcements2 + reinforcements1

        self.run_trials(reinforcements)

        cs, us = 0, 1
        fig, ax = self.plot(cs, us, save=False)
        ax.set_title(f"History of extinction of item {cs} conditioned on item {us}")
        fig.savefig(f"{OUTPUT_DIR}history_of_extinction.png")

    def demonstrate_second_order(self):
        """
        Description: pairing a CS with an already paired CS results in the current CS also
        producing the US.

        Implementation: calculate the association strength by traversing all paths from the
        current CS to the US.

        Validation input: association strength between "light" and "food" begins high. Repeated
        pairings of "bell" and "light" occur.

        Validation output: association strength of "bell" and "food" increases.

        """
        initial_association_strengths = np.eye(3)
        initial_association_strengths[0, 2] = 0.8
        initial_saliences = np.array([0.5, 0.5, self.US_SALIENCE], dtype=np.float)
        initial_uncertainties = np.array([
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ])
        self.initialize(
            initial_association_strengths=initial_association_strengths,
            initial_saliences=initial_saliences,
            initial_uncertainties=initial_uncertainties,
            initialize_default=True,
        )

        stimuli = np.array([1, 1, 0])
        reinforcement = self.Reinforcement(stimuli)
        reinforcements = [reinforcement] * self.DEFAULT_NUM_TRIALS

        self.run_trials(reinforcements)

        cs1, cs2, us = 0, 1, 2
        fig, ax = self.plot(cs1, us, save=False)
        fig, ax = self.plot(cs2, us, save=False, fig=fig, ax=ax)
        ax.set_title(f"Second order condition of item {cs2} on item {us} through item {cs1}")
        fig.savefig(f"{OUTPUT_DIR}second_order.png")

    def demonstrate_preexposure(self):
        """
        Description: pre-exposure to a particular CS without presenting the US results in less
        conditioning.

        Implementation: introduce a temporal aspect of a particular CS salience parameter.

        Validation input: repeated trials of just "light" without "food" occur. Repeated pairings
        of "light" and "food" occur.

        Validation output: The rate at which the association strength increases is smaller than
        in the control.

        """
        self.initialize(initialize_default=True)

        stimuli1 = np.array([1, 0])
        reinforcement1 = self.Reinforcement(stimuli1)
        reinforcements1 = [reinforcement1] * int(self.DEFAULT_NUM_TRIALS * 0.5)

        stimuli2 = np.array([1, 1])
        reinforcement2 = self.Reinforcement(stimuli2)
        reinforcements2 = [reinforcement2] * int(self.DEFAULT_NUM_TRIALS * 0.5)

        reinforcements = reinforcements1 + reinforcements2

        self.run_trials(reinforcements)

        cs, us = 0, 1
        fig, ax = self.plot(cs, us, save=False)

        self.initialize(initialize_default=True)

        stimuli1 = np.array([0, 0])
        reinforcement1 = self.Reinforcement(stimuli1)
        reinforcements1 = [reinforcement1] * int(self.DEFAULT_NUM_TRIALS * 0.5)

        stimuli2 = np.array([1, 1])
        reinforcement2 = self.Reinforcement(stimuli2)
        reinforcements2 = [reinforcement2] * int(self.DEFAULT_NUM_TRIALS * 0.5)

        reinforcements = reinforcements1 + reinforcements2

        self.run_trials(reinforcements)

        fig, ax = self.plot(cs, us, save=False, fig=fig, ax=ax)
        ax.legend([
            f"Preexposure of item {cs} conditioned on item {us}",
            "Control",
        ])
        ax.set_title(f"Pre-exposure of item {cs} conditioned on item {us}")
        fig.savefig(f"{OUTPUT_DIR}preexposure.png")


if __name__ == "__main__":
    main()
