## This is a first draft of the 8-port behavioural model. 
## Potential updates include updating the model to include fitting methods etc.

# We assume that the port is already aligned to the correct port, i.e. theta=0 is the correct port.
# Numbering of the ports will follow clockwise from the correct port, with theta=n*pi/4

import numpy as np

## Global Experiment Constants
N_PORTS = 8
MAX_N_TRIALS = 20
TRIAL_TIME_MAX = 4  # Based on experiment setup
RADIUS = 1 # Arbitrary radius for distance calculations

### Helper functions

def sigmoid(x):
    """Computes the sigmoid function."""
    return np.tanh(x/2)*0.5 + 0.5

def distance(port1, port2):
    """Computes the distance between two ports in the 8-port model."""
    return min(abs(port1 - port2), N_PORTS - abs(port1 - port2))*(np.pi / 4)*RADIUS  # Each port is pi/4 radians apart

class environment:
    """Environment class for the 8-port model. This simulates the environment for a specific recall session,
    up until the first water is received.
        Args:
            correct_port: The correct port to be used in the simulation.
            min_trials: The minimum number of trials until water is available.
            trial_time_max: The maximum time for each trial.
            max_trials: The maximum number of trials.
        """
    def __init__(self,correct_port,min_trials, trial_time_max=TRIAL_TIME_MAX, max_trials = MAX_N_TRIALS, num_ports=N_PORTS):
        self._correct_port = correct_port
        self._min_trials = min_trials # minimum number of trials until water is available
        self.max_trials = max_trials # Arbitrary maximum 
        self.trial_time_max = trial_time_max  # Based on experiment setup
        self.num_ports = num_ports
        self.start_positions = np.arange(self.num_ports)  # Starting positions for the mice
        self.poke_interval = 0.5

    def respondToPoke(self, port, current_trial):
        """Responds to a poke at a given port. Returns True if the poke is at the correct port, False otherwise.
        Args:
            port: The port that was poked.
            current_trial: The current trial number.
        Returns:
            True if the poke is at the correct port, False otherwise."""
        return current_trial >= self._min_trials and port == self._correct_port
        
    

        # Based on the
        # Placeholder for the actual l(t) implementation
        # This should be replaced with the actual logic for the 8-port model

    def timestamp(self, port, previous_port, t_in_trial):
        """Returns the timestamp of the poke and the total time spent in the trial.
        Args:
            port: The port that was poked.
            previous_port: The previous port that was poked.
            current_trial: The current trial number.
            t_sum: The total time spent in the current trial.

        Returns:
            The updated total time spent in the trial.
        t: The timestamp of the poke."""

        alpha = 0.2 #Arbitrary factor for 1/velocity
        t_in_trial += distance(port, previous_port) * alpha + self.poke_interval  # Arbitrary time increment based on distance
        return t_in_trial

    def l(self, stop, current_trial, t_in_trial):
        """Maps the timestap t to its corresponding trial number. Currently uses a
        dual mechanism , one based on confidence of correct port (notes eq.57), and 
        the other on total time spend on a trial (notes eq.47)
        
        Args:
            stop: The stop condition
            current_trial: The current trial number
            t_in_trial: The time spent in the current trial."""

        new_t_in_trial = t_in_trial
        new_current_trial = current_trial
        trial_ended = False
        
        if t_in_trial >= self.trial_time_max:
            trial_ended = True
            new_current_trial += 1  # Increment the trial number
            new_t_in_trial = 0  # Reset the time spent in the trial

        return new_current_trial, new_t_in_trial, trial_ended


class agent:
    """Stateless agent class for the 8-port model. This simulates the agent's behavior in the environment."""
    def __init__(self, rho, sigma):
        self.rho = rho
        self.sigma = sigma 

        self._q1 =  0.98 # Arbitrary value for q1
        self._q2 = 0.98 # Arbitrary value for q2
        self._confidence_threshold = 0.5 # Arbitrary threshold for confidence gate
        self._trial_time_max = 3.8 # Arbitrary threshold for time gate
        self._beta1 = 10 # Strength of confidence_gate firing
        self._beta2 = 10 # Strngth of time_gate firing

    def marginal(self, belief_states):
        """Computes the marginal distribution over ports given the belief states.
        Args:
            belief_states: The current belief states.
        Returns:
            The marginal distribution over ports."""
        return belief_states.sum(axis=1)

    def policy(self, beliefs):
        # Placeholder for the actual policy implementation
        # This should be replaced with the actual logic for the 8-port model
        port = np.argmax(self.marginal(beliefs))  # Choose the port with the highest marginal probability
        # Get the confidence of the chosen port
        confidence = self.marginal(beliefs)[port]
        return (port,confidence)
        
    def update(self, belief_states, port, current_trial):
        """Updates the belief states based on the observed outcome.
        Args:
            belief_states: The current belief states.
            port: The port that was poked.
            current_trial: The current trial number.
        Returns:
            The updated belief states."""
        # Multiplies by likelihood of the observed outcome, using ideal observer.
        new_belief_states = belief_states.copy()
        trial_mask = np.arange(belief_states.shape[1]) <= current_trial
        new_belief_states[port, trial_mask] = 0

        # Normalize the belief states to ensure they sum to 1
        if new_belief_states.sum() != 0:
            new_belief_states /= new_belief_states.sum()

        return new_belief_states

    def _soft_confidence_gate(self, confidence):
        """Determines whether to stop based on the confidence of the chosen port.
        Args:
            confidence: The confidence of the chosen port.
        Returns:
            True if the agent should stop, False otherwise."""
        return sigmoid(self._beta1 * (confidence - self._confidence_threshold))

    def _soft_time_gate(self, t_sum):
        """Determines whether to stop based on the total time spent in the trial.
        Args:
            t_sum: The total time spent in the trial.
        Returns:
            True if the agent should stop, False otherwise."""
        return sigmoid(self._beta2 * (t_sum - self._trial_time_max))

    def stop_prob(self, port_confidence, t_sum):
        """Returns the probability of stopping P_n, Currently uses a
        dual mechanism , one based on confidence of correct port (notes eq.57), and 
        the other on total time spend on a trial (notes eq.47)
        
        Args:
            port_confidence: The confidence of the chosen port.
            t_sum: The total time spent in the trial.

        Returns:
            The probability of stopping."""
        
        return 1-(1-self._q1*self._soft_confidence_gate(port_confidence))*(1-self._q2*self._soft_time_gate(t_sum))


class belief_states:
    
    def __init__(self, f_theta, g_phi, rho=0, sigma=0):
        self.f_theta = f_theta #probability dist as a vector
        self.g_phi = g_phi #probability dist as a vector
        self.state = self.initialize(self.f_theta, self.g_phi)

    @property
    def f_theta(self):
        return self._f_theta

    @f_theta.setter
    def f_theta(self, distribution):
        """Sets the f_theta distribution and checks if it is valid."""
        if distribution.shape != (N_PORTS,1):
            raise ValueError("f_theta must be a numpy array of size 1*8")
        if not np.allclose(distribution.sum(), 1):
            raise ValueError("f_theta must be a valid probability distribution")
        self._f_theta = distribution

    @property
    def g_phi(self):
        return self._g_phi

    @g_phi.setter
    def g_phi(self, distribution):
        """Sets the g_phi distribution and checks if it is valid."""
        if distribution.shape != (MAX_N_TRIALS,1):
            raise ValueError("g_phi must be a numpy array of size 1*20, over the 20 trials")
        if not np.allclose(distribution.sum(), 1):
            raise ValueError("g_phi must be a valid probability distribution")
        self._g_phi = distribution

    def _initialize(self, f_theta, g_phi):
        return np.outer(f_theta, g_phi)

    def update(self, new_state):
        self.state = new_state

    def get_state(self):
        return self.state


class simulation:
    def __init__(self, correct_port, min_trials, rho, sigma, f_theta, g_phi):
        self.environment = environment(correct_port, min_trials)
        self.agent = agent(rho, sigma)
        self.belief_states = belief_states(f_theta, g_phi, self.agent.rho, self.agent.sigma)
        self.sim_data = [] # Outcome of the mice. 

    def run(self):
        self.belief_states = belief_states(self.belief_states.f_theta, self.belief_states.g_phi, self.agent.rho, self.agent.sigma)  # Reset belief states for each run
        self.sim_data = []  # Reset simulation data for each run
        current_trial = 0
        t_in_trial = 0  # Initialize time spent in the current 
        max_pokes = (self.environment.trial_time_max//self.environment.poke_interval +1)*self.environment.max_trials  # Arbitrary limit on the number of pokes to prevent infinite loops

        """Run the simulation. Currently returns a list of tuples (poke, trial) """
        for poke_count in range(int(max_pokes)):
            port_to_poke, confidence = self.agent.act(self.belief_states.get_state()) #Port and confidence for p_t+1 given current state of beliefs  # Check if the agent received water
            t_sum = self.environment.timestamp(port_to_poke, self.sim_data[-1][0] if self.sim_data and t_in_trial > 0 else None , t_in_trial)  # Update time spent in the trial
            current_trial, t_in_trial, trial_ended = self.environment.l((port_to_poke, confidence), current_trial, t_sum)
            
            self.sim_data.append((port_to_poke, current_trial, t_sum))
            if current_trial >= self.environment.max_trials:  # Stop the simulation if water is received, max trials reached, or trial ended
                break
            w_t = self.environment.respondToPoke(port_to_poke, current_trial)
            
            self.belief_states.update(self.agent.update(self.belief_states.get_state(), port_to_poke, current_trial))  # Update the agent's beliefs based on the observed outcome
            stop_prob = self.agent.stop_prob(confidence, t_sum)  # Get the probability of stopping
            if np.random.rand() < stop_prob:  # Stop the simulation based on the stopping probability
                current_trial+=1
                t_in_trial = 0
        print("Simulation completed. Total trials: ", current_trial)
        return self.sim_data

# In practice, l(t) can be found directly from the data. Here we w

sim = simulation(correct_port=0, min_trials=3, rho=0.5, sigma=0.1, f_theta=np.array([[0.125]*8]).T, g_phi=np.array([[0.05]*20]).T)