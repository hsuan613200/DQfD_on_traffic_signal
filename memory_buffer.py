# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:50:58 2020

@author: HP
"""

import random
import numpy as np

from collections import deque
from sumtree import SumTree

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per = False):
        """ Initialization
        """
        if(with_per):
            # Prioritized Experience Replay
            self.alpha = 0.5
            self.epsilon = 0.01
            self.abs_err_upper = 1
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self.count = 0
        self.with_per = with_per
        self.buffer_size = buffer_size

    def memorize(self, state, phase_state, action, reward, next_state, next_phase_state, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """

        experience = (state, phase_state, action, reward, next_state, next_phase_state)
        if(self.with_per):
            max_p = np.max(self.buffer.tree[-self.buffer.capacity:])
            if max_p == 0:
                max_p = self.abs_err_upper
            priority = self.priority(max_p)
            self.buffer.add(priority, experience)
            self.count += 1
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample using prorities
        if(self.with_per):
            T = self.buffer.total() // batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                idx, error, data = self.buffer.get(s)
                batch.append((*data, idx))
            idx = np.array([i[6] for i in batch])
        # Sample randomly from Buffer
        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        all_state = {}
        all_next_state = {}
        
        for pos in ["North", "West", "South", "East"]:
            all_state[pos + '_input'] = np.array([i[0][pos] for i in batch])
            all_next_state[pos + '_input'] = np.array([i[4][pos] for i in batch])
        a_batch = np.array([i[2] for i in batch])
        r_batch = np.array([i[3] for i in batch])
        all_state['phase_input'] = np.array([i[1] for i in batch])
        all_next_state['phase_input'] = np.array([i[5] for i in batch])
        
        
        return all_state, a_batch, r_batch, all_next_state, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        error = self.priority(new_error)
        clipped_error = np.minimum(error, self.abs_err_upper)
        self.buffer.update(idx, clipped_error)

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(self.buffer_size)
        else: self.buffer = deque()
        self.count = 0