# Block-BFGS Optimizer for ASE

This repository provides a custom optimizer for the Atomistic Simulation Environment (ASE), implementing a **Block-BFGS (B-BFGS)** algorithm.

This optimizer inherits from the standard `ase.optimize.bfgs.BFGS` class but uses a block-based Hessian update strategy instead of the traditional rank-2 update. The implementation is contained in a single file, `block_bfgs.py`.

## Key Features & Performance

The standard BFGS algorithm performs a rank-2 update to the approximate Hessian matrix at each step. In contrast, this Block-BFGS optimizer uses a block of the $q$ most recent steps and gradient differences to perform a more robust rank-2q update.

The primary advantage of this approach is its potential for **improved convergence performance and stability**, especially for large-scale systems, complex molecules, or noisy potential energy surfaces. By incorporating more historical information into each update, the optimizer can build a more accurate model of the Hessian, potentially leading to faster convergence in fewer iterations compared to standard BFGS.

The block update logic is based on methods described in computational chemistry and optimization literature (e.g., [arXiv:1609.00318](https://arxiv.org/pdf/1609.00318)).

## Requirements

* **Python 3.12**
* **ASE (Atomistic Simulation Environment)**
    * This module has been tested with **ASE version 3.26.0**.
* **NumPy 2.2.6**

## Installation

Since this is a single-file module, you can directly download `block_bfgs.py` and place it in your project's working directory or in a location included in your `PYTHONPATH`.

## Usage Example

The `BlockBFGS` optimizer can be used as a drop-in replacement for any other ASE optimizer, such as `BFGS` or `LBFGS`.

Here is a minimal example of relaxing a water molecule using an EMT calculator:

```python
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT

# Import the custom optimizer from the local file
from block_bfgs import BlockBFGS

# 1. Set up your Atoms object and Calculator
atoms = molecule('H2O')
atoms.calc = EMT()

# 2. Instantiate the BlockBFGS optimizer
# You can customize the block_size and max_window
dyn = BlockBFGS(
    atoms,
    logfile='h2o_opt.log',
    trajectory='h2o_opt.traj',
    block_size=4,  # Use 4 recent steps for the update
    max_window=8   # Store up to 8 steps in history
)

# 3. Run the optimization
print("Running optimization with BlockBFGS...")
try:
    dyn.run(fmax=0.0001)
    print("Optimization finished successfully.")
except Exception as e:
    print(f"Optimization failed: {e}")

# 4. Print final positions
print("Final positions:")
symbols = atoms.get_chemical_symbols()
positions = atoms.get_positions()

for symbol, pos in zip(symbols, positions):
    print(f"{symbol:<4} {pos[0]:12.8f} {pos[1]:12.8f} {pos[2]:12.8f}")

```

## Key Parameters

In addition to the standard ASE optimizer parameters, BlockBFGS accepts two new arguments:block_size (int, default: 4): The number of recent steps ($q$) to use in the block update.max_window (int, default: 8): The maximum number of (step, gradient_difference) pairs to store in history. This value must be greater than or equal to block_size.

## License

This project is licensed under the GNU Lesser General Public License, Version 2.1 (LGPL-2.1). See the LICENSE file (or COPYING.LESSER) for details.


