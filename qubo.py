import numpy as np
from qiskit_aer import Aer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from typing import Dict, List, Tuple, Union
from scipy.sparse import spmatrix, csr_matrix

class QUBOSolver:
    """
    A class for solving QUBO problems using Qiskit's quantum algorithms.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        # Set random seed for numpy which is used by the algorithms
        np.random.seed(seed)
        
    def create_qubo_model(self, Q: Union[np.ndarray, spmatrix], constant: float = 0.0) -> QuadraticProgram:
        """
        Create a QUBO model from a Q matrix.
        
        Args:
            Q: The QUBO matrix (symmetric), can be sparse or dense
            constant: Constant term in the objective function
            
        Returns:
            A QuadraticProgram object representing the QUBO problem
        """
        # Convert Q to CSR format if it's sparse, or to dense array if it's not
        if isinstance(Q, spmatrix):
            Q = Q.tocsr()
            num_vars = Q.shape[0]
        else:
            Q = (Q + Q.T) / 2  # Ensure Q is symmetric if dense
            num_vars = Q.shape[0]
        
        # Create a quadratic program
        qubo = QuadraticProgram(name="QUBO Problem")
        
        # Add binary variables
        for i in range(num_vars):
            qubo.binary_var(name=f'x{i}')
        
        # Set the objective with the Q matrix
        linear = {}
        quadratic = {}
        
        # Handle diagonal elements (linear terms)
        if isinstance(Q, spmatrix):
            for i in range(num_vars):
                val = float(Q[i, i])
                if abs(val) > 1e-10:  # Use small threshold for numerical stability
                    linear[f'x{i}'] = val
            
            # Handle off-diagonal elements (quadratic terms)
            rows, cols = Q.nonzero()
            for i, j in zip(rows, cols):
                if i < j:  # Only upper triangle
                    val = float(Q[i, j])
                    if abs(val) > 1e-10:  # Use small threshold for numerical stability
                        quadratic[(f'x{i}', f'x{j}')] = 2 * val  # Double the interaction term
        else:
            for i in range(num_vars):
                val = float(Q[i, i])
                if abs(val) > 1e-10:
                    linear[f'x{i}'] = val
            
            for i in range(num_vars):
                for j in range(i+1, num_vars):
                    val = float(Q[i, j])
                    if abs(val) > 1e-10:
                        quadratic[(f'x{i}', f'x{j}')] = 2 * val  # Double the interaction term
        
        # Create the quadratic program with the objective function
        qubo.minimize(linear=linear, quadratic=quadratic, constant=constant)
        return qubo
        
    def solve_qubo_classical(self, Q: Union[np.ndarray, spmatrix], constant: float = 0.0) -> Tuple[Dict[str, int], float]:
        """
        Solve a QUBO problem using classical optimization.
        
        Args:
            Q: The QUBO matrix, can be sparse or dense
            constant: Constant term in the objective function
            
        Returns:
            Tuple of (solution, objective_value)
        """
        qubo = self.create_qubo_model(Q, constant)
        
        # Use NumPyMinimumEigensolver with exact solver
        exact_mes = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(exact_mes)
        
        # Solve and ensure binary solution
        result = optimizer.solve(qubo)
        
        # Convert numpy array solution to dictionary format
        if isinstance(result.x, np.ndarray):
            solution = {f'x{i}': int(round(val)) for i, val in enumerate(result.x)}
        else:
            solution = {var_name: int(round(float(value))) for var_name, value in result.x.items()}
        
        return solution, result.fval
    
    def solve_qubo_quantum(self, Q: np.ndarray, constant: float = 0.0, p: int = 1, shots: int = 1024) -> Tuple[Dict[str, int], float]:
        """
        Solve a QUBO problem using QAOA.
        
        Args:
            Q: The QUBO matrix
            constant: Constant term in the objective function
            p: QAOA parameter (number of layers)
            shots: Number of shots for the quantum circuit
            
        Returns:
            Tuple of (solution, objective_value)
        """
        qubo = self.create_qubo_model(Q, constant)
        
        # Set up the quantum backend and components needed for QAOA
        sampler = Sampler()
        optimizer = COBYLA()
        
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(qubo)
        
        return result.x, result.fval
    
    def qubo_from_ising(self, h: np.ndarray, J: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        Convert an Ising model to a QUBO matrix.
        
        Args:
            h: The linear coefficients of the Ising model
            J: The quadratic coefficients of the Ising model
            
        Returns:
            The QUBO matrix Q
        """
        n = len(h)
        Q = np.zeros((n, n))
        
        # Set diagonal terms
        for i in range(n):
            Q[i, i] = 2 * h[i]
            
        # Set off-diagonal terms
        for (i, j), value in J.items():
            Q[i, j] = 4 * value
            Q[i, i] -= 2 * value
            Q[j, j] -= 2 * value
            
        return Q
    
    def format_solution(self, solution: Dict[str, int], n: int = None) -> np.ndarray:
        """
        Format the solution as a numpy array.
        
        Args:
            solution: The solution dictionary from the solver
            n: The expected length of the solution array
            
        Returns:
            A numpy array containing the binary solution
        """
        if isinstance(solution, dict):
            # If n is not specified, infer from the solution keys
            if n is None:
                n = max([int(k[1:]) for k in solution.keys()]) + 1
            
            result = np.zeros(n, dtype=int)
            for key, value in solution.items():
                idx = int(key[1:])  # Extract index from key like 'x0', 'x1', etc.
                # Convert value to binary (0 or 1) - fix for cases where value might be a float
                result[idx] = int(round(float(value)))
            return result
        else:
            # If solution is already a numpy array or list
            return np.array(solution, dtype=int)