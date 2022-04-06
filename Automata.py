import numpy as np
from scipy.signal import convolve


class Automata:
    ''' Define and evaluate Larger-Than-Life style cellular automata rules.
        Supports grids of arbitrary dimension.
        Also suppoorts a 'generalized' mode which allows for custom non-totalistic transition functions.
    '''

    def __init__(self, rule, edge_mode='WALL', rule_format='standard'):
        self.edge_mode = None
        self.set_rule(rule, rule_format)
        self.set_edge_mode(edge_mode)

    ### Main ###

    def set_initial_state(self, init_state):
        # Load init_state into the instance and set up the appropriate kernel
        if (self.grid is None or init_state.shape != self.grid.shape):
            self.grid = init_state
            if self._format == 0:
                # Create a convolution kernel which sums all values in radius sized n-d neighborhood
                dimension = len(init_state.shape)
                self._build_standard_kernel(dimension)
                # Align rules arrays to the correct number of dimensions
                self.rule['Ss'] = self.rule['Ss'].reshape(-1, *[1]*dimension)
                self.rule['Bb'] = self.rule['Bb'].reshape(-1, *[1]*dimension)
            else:
                # Create an index array to concatenate each neghborhood into a vector
                self._build_generalized_kernel(init_state.shape)
        else:
            self.grid = init_state

    def evolve_grid(self, steps=1, history_window=0):
        # Evolve for a number of steps.
        # Return up to history_window past generations (-1 to return full history)
        if self.grid is None:
            return None

        if history_window == -1 or history_window >= steps:
            history_window = steps
        history_window += 1

        if self._format == 0:
            step_func = self._apply_step_standard
        else:
            step_func = self._apply_step_generalized

        if history_window == 1 and steps == 1:
            step_func()
            return self.grid

        history = np.zeros([history_window]+list(self.grid.shape), dtype=int)
        history[0] = self.grid
        for step in range(1, steps+1):
            step_func()
            if step < history_window:
                history[step] = self.grid
            else:
                np.roll(history, -1, axis=0)
                history[history_window-1] = self.grid
        return history

    ### Property setters ###

    def set_rule(self, rule, format):
        # Set automata rule for instance. See documentation for format and options
        rule.setdefault('Rr', 1)
        rule['Rr'] = int(rule['Rr'])
        if (rule['Rr'] < 0):
            raise ValueError(f' \'Rr\' value must be whole number')

        rule.setdefault('Cc', 2)
        rule['Cc'] = int(rule['Cc'])
        if (rule['Cc'] < 0):
            raise ValueError(f' \'Cc\' value must be whole number')
        if rule['Cc'] < 2:
            rule['Cc'] = 2

        rule.setdefault('Mm', 0)
        rule['Mm'] = bool(rule['Mm'])

        rule.setdefault('Nn', 'M')
        if rule['Nn'] not in ['M', 'N']:
            raise ValueError(
                f'Invalid neighborhood type.\n Accepted values are \'M\' or \'N\'')

        if format == 'standard':
            if ((rule.get('Ss') is None) or (rule.get('Bb') is None)):
                raise ValueError(
                    'Standard rule format requires \'Ss\' and \'Bs\' keys')
            rule['Ss'] = np.asarray(rule['Ss'])
            rule['Bb'] = np.asarray(rule['Bb'])
            self._format = 0
        elif format == 'generalized':
            if (rule.get('state_func') is None):
                raise ValueError(
                    'Generalized rule format requires \'state_func\' key pair')
            self._format = 1
        else:
            raise ValueError(
                f'Unexpected rule format descriptor: \'{format}\'')

        if self.edge_mode is not None:
            self.set_edge_mode(self.edge_mode)
        self.rule = rule
        self.grid = None

    def set_edge_mode(self, edge_mode):
        # Set edge rule. Rule format documentation for options
        self.edge_mode = edge_mode
        radius = self.rule['Rr']

        def grow_padder(state):
            cells = np.nonzero(state)
            upper_bounds = [size - radius for size in state.shape]
            for axis in range(len(cells)):
                if np.amax(cells[axis]) >= upper_bounds[axis] or np.amin(cells[axis]) <= radius:
                    self.grid = np.pad(
                        self.grid, pad_width=radius, constant_values=0)
                    return np.pad(state, pad_width=2*radius, constant_values=0)

            return np.pad(state, pad_width=radius, constant_values=0)

        if self.edge_mode == "WALL":
            self._padder = lambda state: np.pad(
                state, pad_width=radius, constant_values=0)
        elif self.edge_mode == "WRAP":
            self._padder = lambda state: np.pad(
                state, pad_width=radius, mode='wrap')
        elif self.edge_mode == "GROW":
            self._padder = grow_padder
        else:
            raise ValueError(
                "edge_mode supports arguments: {\"WALL\", \"WRAP\", \"GROW\"}")

    ### Internal ###

    def _apply_step_standard(self):
        # Get living cells and pad according to edge mode
        padded_state = self._padder((self.grid == 1).astype(int))
        # Convolve to count all neighbors
        out = convolve(padded_state, self.kernel, mode='valid')
        # Add dimension to ensure alignment when counts are compared
        adj_counts = np.expand_dims(out, axis=0)
        # Construct boolean mask for cells that should not survive this step
        age_fits = np.all(adj_counts != self.rule['Ss'], axis=0)
        # Age all nonzero cells where age_fits is true
        self.grid[np.logical_and(age_fits, self.grid != 0)] += 1
        # Set all birth cells to 1.
        birth_fits = np.any(adj_counts == self.rule['Bb'], axis=0)
        self.grid[np.logical_and(birth_fits, self.grid == 0)] = 1
        # Kill cells that are beyond the max possible state
        self.grid[self.grid == self.rule['Cc']] = 0

    def _apply_step_generalized(self):
        # Pad the grid according to edge mode
        padded_state = self._padder(self.grid)
        if (self.edge_mode == 'GROW' and padded_state.shape != self.grid.shape):
            new_grid_shape = tuple([d - 2 * self.rule['Rr']
                                   for d in padded_state.shape])
            self._build_generalized_kernel(new_grid_shape)
        vector_grid = padded_state[self.kernel]
        self.grid = self.rule['state_func'](vector_grid)

    def _build_standard_kernel(self, dimension):
        radius = self.rule['Rr']
        kernel_shape = [2*radius + 1] * dimension
        if self.rule['Nn'] == 'M':
            # Build Moore neighborhood (n-d hypercube of 1s)
            self.kernel = np.ones(kernel_shape, dtype=int)
        elif self.rule['Nn'] == 'N':
            # Build Von Neumann neighborhood (compute L1 distance from kernel center for every index, and set elements by radius threshold to 0 or 1)
            inds = np.abs(np.indices(kernel_shape) - radius)
            dists = np.sum(inds, axis=0)
            self.kernel = np.where(dists <= radius, 1, 0)
        # Set if middle cell is included in neighbor count
        self.kernel[tuple([radius] * dimension)] = self.rule['Mm']

    def _build_generalized_kernel(self, shape):
        # Not a convolution kernel as in standard format. Generates an indices tuple which produces neighborhood vectors per grid element
        radius = self.rule['Rr']
        dimension = len(shape)
        # Sparse coordinates for each grid point
        centers = np.indices(shape, sparse=True)
        # Sparse coordinate offsets defining the neighborhood
        k_grid = np.indices([2*radius+1] * dimension, sparse=True)
        indices = []
        for i in range(dimension):
            # Add axes to align the center and k_grid dimensions
            center = centers[i].reshape(
                centers[i].shape + tuple([1] * dimension))
            k = k_grid[i].reshape(tuple([1] * dimension) + k_grid[i].shape)
            # Shift values to range (-radius, radius)
            center += radius
            k -= radius
            # Add indices to create full index matrix for axis i
            indices.append(center+k)

        self.kernel = tuple(indices)

    def build_self_sim(self, size):
        # return self similarity matrix for a size step evaluation
        path = self.evolve_grid(size, -1)
        sim_array = np.zeros((size, size))
        for j in range(size):
            for i in range(size):
                sim_array[i, j] = np.mean(np.square(path[i]-path[j]))
        return sim_array
