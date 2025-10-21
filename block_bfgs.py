# fmt: off

import warnings
from pathlib import Path
from typing import IO, Optional, Union

import numpy as np
from numpy.linalg import eigh

from ase import Atoms
from ase.optimize.optimize import Optimizer, UnitCellFilter
# Import the BFGS class from bfgs.py
from ase.optimize.bfgs import BFGS

# Reference for block update logic:
# https://arxiv.org/pdf/1609.00318



def symm(A):
    return 0.5 * (A + A.T)

def safe_inv(A, reg=1e-10):
    """Invert A with small regularization fallback, then pinv."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Areg = A + reg * np.eye(A.shape[0])
        try:
            return np.linalg.inv(Areg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(Areg)

class BlockHessianUpdate:
    def __init__(self, block_size=4, max_window=8, denom_threshold=1e-12, inv_reg=1e-10):
        """
        block_size: number of stored steps to use when performing a block update
        max_window: maximum history length to retain (>= block_size)
        """
        assert max_window >= block_size
        self.block_size = int(block_size)
        self.max_window = int(max_window)
        self.denom_threshold = denom_threshold
        self.inv_reg = inv_reg

        # history stored as lists of vectors (each vector shape (n,))
        self.S_list = []
        self.Y_list = []

    def delete_old_data(self):
        """Drop the oldest history item (if any)."""
        if self.S_list:
            self.S_list.pop(0)
            self.Y_list.pop(0)

    def _push_history(self, s, y):
        """Append new step (s,y), maintain window."""
        self.S_list.append(s.copy())
        self.Y_list.append(y.copy())
        if len(self.S_list) > self.max_window:
            self.S_list.pop(0); self.Y_list.pop(0)

    def _assemble_block(self, use_last_k=None):
        """Return S (n x q) and Y (n x q) matrices from most recent columns."""
        if use_last_k is None:
            use_last_k = min(self.block_size, len(self.S_list))
        k = min(use_last_k, len(self.S_list))
        if k == 0:
            return None, None
        # take last k entries
        Scols = [self.S_list[-k + i] for i in range(k)]
        Ycols = [self.Y_list[-k + i] for i in range(k)]
        S = np.column_stack(Scols)   # n x k
        Y = np.column_stack(Ycols)
        return S, Y


    def _block_BFGS_update(self, B, S, Y):
        """
        B <- B - B S (S^T B S)^{-1} S^T B + Y (S^T Y)^{-1} Y^T
        S,Y are n x q with columns as steps.
        """
        if S is None or Y is None:
            return B.copy()
        # filter near linear dependence in S by SVD (drop tiny singular values)
        U, svals, Vt = np.linalg.svd(S, full_matrices=False)
        keep = svals > 1e-8
        if not np.any(keep):
            return B.copy()
        # choose columns corresponding to largest contributions
        rank = np.sum(keep)
        col_norms = np.linalg.norm(S, axis=0)
        idx_sorted = np.argsort(-col_norms)
        keep_idx = np.sort(idx_sorted[:rank])
        Sf = S[:, keep_idx]
        Yf = Y[:, keep_idx]

        M1 = np.dot(np.dot(Sf.T, B), Sf)       # q x q
        M2 = np.dot(Sf.T, Yf)              # q x q

        invM1 = safe_inv(M1, reg=self.inv_reg)
        invM2 = safe_inv(M2, reg=self.inv_reg)

        term1 = np.dot(np.dot(np.dot(B, Sf), invM1), np.dot(Sf.T, B))
        term2 = np.dot(np.dot(Yf, invM2), Yf.T)
        Bp = B - term1 + term2
        return symm(Bp)



class BlockBFGS(BFGS):
    """Block BFGS optimizer.

    This optimizer inherits from ASE's BFGS but uses a block Hessian update
    strategy based on the BlockHessianUpdate class.
    
    This version is hard-coded to *only* use the block_BFGS update rule.
    """
    
    # Add block update defaults to BFGS.defaults
    defaults = {
        **BFGS.defaults,
        'block_size': 4,
        'max_window': 8,
    }

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Optional[Union[IO, str, Path]] = '-',
        trajectory: Optional[Union[str, Path]] = None,
        append_trajectory: bool = False,
        maxstep: Optional[float] = None,
        alpha: Optional[float] = None,
        block_size: Optional[int] = None,
        max_window: Optional[int] = None,
        **kwargs,
    ):
        """Block BFGS optimizer (block_BFGS method only).

        Parameters
        ----------
        atoms: :class:`~ase.Atoms`
            The Atoms object to relax.
        
        ... (BFGS parameter descriptions omitted) ...

        block_size: int
            Number of stored steps to use when performing a block update.
            (Default: 4)

        max_window: int
            Maximum history length to retain (>= block_size).
            (Default: 8)
        
        """
        
        # Get block parameters from kwargs or use default values
        if block_size is None:
            self.block_size = kwargs.pop('block_size', self.defaults['block_size'])
        else:
            kwargs.pop('block_size', None) # Remove from kwargs if present
            self.block_size = block_size
            
        if max_window is None:
            self.max_window = kwargs.pop('max_window', self.defaults['max_window'])
        else:
            kwargs.pop('max_window', None)
            self.max_window = max_window

        if self.max_window < self.block_size:
            warnings.warn(f"max_window ({self.max_window}) is less than "
                          f"block_size ({self.block_size}). "
                          f"Setting max_window = block_size.")
            self.max_window = self.block_size

        # Initialize BlockHessianUpdate instance
        self.block_updater = BlockHessianUpdate(
            block_size=self.block_size,
            max_window=self.max_window
        )

        # Call the parent class (BFGS) __init__
        super().__init__(
            atoms=atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            maxstep=maxstep,
            alpha=alpha,
            **kwargs,  # Pass the remaining kwargs
        )


    def initialize(self):
        """Initialize the optimizer.
        
        Calls the parent (BFGS) initialize and also clears the
        history of the block updater.
        """
        # Call parent class's initialize (sets H0, H=None, etc.)
        super().initialize()
        
        # Reset the block updater's history
        self.block_updater.S_list = []
        self.block_updater.Y_list = []

    def update(self, pos, forces, pos0, forces0):
        """Update the Hessian matrix using the block_BFGS method.
        
        Parameters:
        pos, forces: flat arrays of current positions and forces (gradients).
        pos0, forces0: flat arrays of previous positions and forces.
        """
        
        if self.H is None:
            self.H = self.H0
            return
        
        s = pos - pos0  # displacement (s)
        
        if np.abs(s).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        y = forces - forces0  # gradient difference (y)
        
        # 1. Add the new (s, y) pair to the history
        self.block_updater._push_history(s, y)
        
        # 2. Assemble the S and Y matrices for the block update
        S, Y = self.block_updater._assemble_block(use_last_k=self.block_size)
        
        # 3. If history is sufficient, update Hessian directly with block_BFGS method
        if S is not None:
            # Hard-code _block_BFGS_update
            self.H = self.block_updater._block_BFGS_update(self.H, S, Y)