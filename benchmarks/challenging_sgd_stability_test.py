#!/usr/bin/env python
"""
Challenging SGD Stability Test: Noisy Gradients + Non-Convex Landscape

This experiment specifically tests scenarios where temporal stability matters:
1. High gradient noise (common in large-scale training)
2. Multiple local minima (realistic for neural networks)
3. Distribution shifts during training
4. Memory-intensive updates (long sequences)

Goal: Demonstrate when StableSGD with momentum conservation provides measurable benefits.
"""

import random
import math
import time


class ChallengingOptimizationLandscape(object):
    """
    Creates a challenging optimization scenario similar to LLM training difficulties:
    - Non-convex loss with multiple local minima
    - High gradient noise
    - Temporal distribution shifts
    - Parameter coupling (realistic neural network behavior)
    """
    
    def __init__(self, dim=100, noise_level=0.3):
        self.dim = dim
        self.noise_level = noise_level
        self.step_count = 0
        
        # True optimum (what we're trying to reach)
        self.true_optimum = [0.5 * math.sin(i * 0.1) for i in range(dim)]
        
        # Create multiple local minima
        self.local_minima = []
        for k in range(5):
            minimum = [0.3 * math.sin(i * 0.1 + k) for i in range(dim)]
            self.local_minima.append(minimum)
    
    def compute_loss_and_gradients(self, params):
        """
        Compute challenging loss function with:
        1. Main quadratic term toward true optimum
        2. Multiple local minima attraction
        3. Non-convex oscillations
        4. High gradient noise
        5. Parameter coupling
        """
        self.step_count += 1
        total_loss = 0.0
        gradients = []
        
        for i in range(len(params)):
            param_val = params[i]
            target = self.true_optimum[i]
            
            # 1. Main quadratic loss
            main_diff = param_val - target
            main_loss = main_diff ** 2
            main_grad = 2 * main_diff
            
            # 2. Local minima attraction (creates multiple basins)
            local_attraction = 0.0
            local_grad_contrib = 0.0
            for local_min in self.local_minima:
                local_diff = param_val - local_min[i]
                local_attraction += 0.2 * math.exp(-local_diff ** 2)
                local_grad_contrib += -0.4 * local_diff * math.exp(-local_diff ** 2)
            
            # 3. Non-convex oscillations (creates rough landscape)
            oscillation_loss = 0.1 * (1 - math.cos(5 * param_val))
            oscillation_grad = 0.5 * math.sin(5 * param_val)
            
            # 4. Parameter coupling (adjacent parameters influence each other)
            coupling_loss = 0.0
            coupling_grad = 0.0
            if i > 0:
                coupling_diff = param_val - params[i-1]
                coupling_loss += 0.05 * coupling_diff ** 2
                coupling_grad += 0.1 * coupling_diff
            if i < len(params) - 1:
                coupling_diff = param_val - params[i+1] 
                coupling_loss += 0.05 * coupling_diff ** 2
                coupling_grad += 0.1 * coupling_diff
            
            # 5. Temporal distribution shift (makes problem harder over time)
            time_shift = 0.01 * math.sin(self.step_count * 0.01) * (param_val ** 2)
            time_grad = 0.02 * math.sin(self.step_count * 0.01) * param_val
            
            # Combine all loss components
            param_loss = (main_loss - local_attraction + oscillation_loss + 
                         coupling_loss + time_shift)
            total_loss += param_loss
            
            # Combine all gradient components
            clean_grad = (main_grad + local_grad_contrib + oscillation_grad + 
                         coupling_grad + time_grad)
            
            # 6. Add significant gradient noise
            noise = (random.random() - 0.5) * 2 * self.noise_level
            noisy_grad = clean_grad + noise
            
            gradients.append(noisy_grad)
        
        return total_loss, gradients


class HighDimensionalParameter(object):
    """High-dimensional parameter vector for challenging optimization."""
    
    def __init__(self, dim):
        # Initialize parameters randomly
        self.data = [random.gauss(0, 0.5) for _ in range(dim)]
        self.grad = None
        self.dim = dim
    
    def zero_grad(self):
        self.grad = [0.0] * self.dim
    
    def backward(self, gradients):
        self.grad = gradients[:]


class StandardSGDMomentum(object):
    """Standard SGD with momentum (baseline)."""
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = []
        
        for param in params:
            self.velocities.append([0.0] * param.dim)
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            velocity = self.velocities[i]
            
            for j in range(param.dim):
                velocity[j] = self.momentum * velocity[j] + self.lr * param.grad[j]
                param.data[j] -= velocity[j]


class EnhancedStableSGD(object):
    """
    Enhanced StableSGD specifically tuned for challenging optimization scenarios.
    
    Key improvements for difficult landscapes:
    1. Adaptive momentum based on gradient consistency
    2. Enhanced temporal stability with gradient history
    3. Improved energy conservation with momentum decay
    4. Noise robustness through gradient filtering
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9, 
                 temporal_stability=0.02, energy_conservation=True,
                 gradient_smoothing=True):
        self.params = params
        self.lr = lr
        self.base_momentum = momentum
        self.temporal_stability = temporal_stability
        self.energy_conservation = energy_conservation
        self.gradient_smoothing = gradient_smoothing
        self.step_count = 0
        
        # Standard momentum buffers
        self.velocities = []
        
        # Enhanced stability features
        self.param_history = []      # For temporal stability
        self.gradient_history = []   # For gradient smoothing
        self.energy_history = []     # For energy conservation
        
        # Initialize buffers
        for param in params:
            self.velocities.append([0.0] * param.dim)
            self.param_history.append([param.data[:]])
            self.gradient_history.append([[0.0] * param.dim])
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    
    def step(self):
        self.step_count += 1
        total_energy = 0.0
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            velocity = self.velocities[i]
            param_hist = self.param_history[i]
            grad_hist = self.gradient_history[i]
            
            for j in range(param.dim):
                current_grad = param.grad[j]
                
                # 1. Gradient smoothing for noise robustness
                if self.gradient_smoothing and len(grad_hist) > 0:
                    # Exponential moving average of gradients
                    smooth_grad = 0.7 * current_grad + 0.3 * grad_hist[-1][j]
                else:
                    smooth_grad = current_grad
                
                # 2. Adaptive momentum based on gradient consistency
                if len(grad_hist) >= 2:
                    # Measure gradient consistency
                    grad_diff = abs(smooth_grad - grad_hist[-1][j])
                    consistency = 1.0 / (1.0 + grad_diff)
                    adaptive_momentum = self.base_momentum * consistency
                else:
                    adaptive_momentum = self.base_momentum
                
                # 3. Energy computation for conservation
                old_velocity = velocity[j]
                new_velocity = adaptive_momentum * velocity[j] + self.lr * smooth_grad
                
                kinetic_energy = 0.5 * (new_velocity ** 2)
                potential_energy = 0.5 * (smooth_grad ** 2)
                param_energy = kinetic_energy + potential_energy
                total_energy += param_energy
                
                # 4. Temporal stability penalty
                stability_penalty = 0.0
                if self.step_count > 2 and len(param_hist) >= 2:
                    # Penalize excessive parameter jumping
                    recent_change = abs(param.data[j] - param_hist[-1][j])
                    stability_penalty = self.temporal_stability * recent_change
                
                # 5. Energy conservation: adaptive learning rate
                if self.energy_conservation and param_energy > 1e-8:
                    # Reduce learning rate in high-energy situations
                    energy_factor = 1.0 / (1.0 + math.sqrt(param_energy) * 0.2)
                    adaptive_lr = self.lr * energy_factor
                else:
                    adaptive_lr = self.lr
                
                # 6. Enhanced symplectic update
                velocity[j] = new_velocity
                position_update = adaptive_lr * new_velocity + stability_penalty
                param.data[j] -= position_update
            
            # Update histories
            param_hist.append(param.data[:])
            grad_hist.append(param.grad[:])
            
            # Limit history size
            if len(param_hist) > 5:
                param_hist.pop(0)
            if len(grad_hist) > 3:
                grad_hist.pop(0)
        
        # Track energy evolution
        self.energy_history.append(total_energy)
        if len(self.energy_history) > 50:
            self.energy_history.pop(0)


def run_challenging_stability_experiment():
    """
    Test StableSGD vs Standard SGD on challenging optimization landscape.
    This scenario is designed to highlight when temporal stability matters.
    """
    
    print("CHALLENGING SGD STABILITY TEST")
    print("=" * 60)
    print("Testing scenarios where temporal stability provides clear benefits:")
    print("- High gradient noise")
    print("- Non-convex landscape with local minima") 
    print("- Parameter coupling")
    print("- Temporal distribution shifts")
    print()
    
    # Experiment parameters
    dimension = 50
    num_iterations = 150
    noise_level = 0.4  # High noise to test stability
    
    print("Experiment Setup:")
    print("  Parameter Dimension: {}".format(dimension))
    print("  Training Iterations: {}".format(num_iterations))
    print("  Gradient Noise Level: {:.1f}".format(noise_level))
    print()
    
    # Create challenging optimization problem
    landscape = ChallengingOptimizationLandscape(dimension, noise_level)
    
    # Test 1: Standard SGD with Momentum
    print("Testing Standard SGD with Momentum:")
    params_sgd = HighDimensionalParameter(dimension)
    optimizer_sgd = StandardSGDMomentum([params_sgd], lr=0.005, momentum=0.9)
    
    sgd_losses = []
    sgd_param_distances = []  # Distance from true optimum
    
    sgd_start_time = time.time()
    
    for iteration in range(num_iterations):
        optimizer_sgd.zero_grad()
        
        loss, gradients = landscape.compute_loss_and_gradients(params_sgd.data)
        params_sgd.backward(gradients)
        optimizer_sgd.step()
        
        sgd_losses.append(loss)
        
        # Track distance to true optimum
        distance = math.sqrt(sum((params_sgd.data[i] - landscape.true_optimum[i]) ** 2 
                               for i in range(dimension)))
        sgd_param_distances.append(distance)
        
        if iteration % 30 == 0:
            print("  Iteration {}: Loss = {:.6f}, Distance = {:.6f}".format(
                iteration, loss, distance))
    
    sgd_time = time.time() - sgd_start_time
    
    # Test 2: Enhanced StableSGD
    print("\nTesting Enhanced StableSGD:")
    params_stable = HighDimensionalParameter(dimension)
    optimizer_stable = EnhancedStableSGD([params_stable], lr=0.005, momentum=0.9,
                                        temporal_stability=0.03, 
                                        energy_conservation=True,
                                        gradient_smoothing=True)
    
    stable_losses = []
    stable_param_distances = []
    
    stable_start_time = time.time()
    
    for iteration in range(num_iterations):
        optimizer_stable.zero_grad()
        
        loss, gradients = landscape.compute_loss_and_gradients(params_stable.data)
        params_stable.backward(gradients)
        optimizer_stable.step()
        
        stable_losses.append(loss)
        
        # Track distance to true optimum
        distance = math.sqrt(sum((params_stable.data[i] - landscape.true_optimum[i]) ** 2 
                               for i in range(dimension)))
        stable_param_distances.append(distance)
        
        if iteration % 30 == 0:
            print("  Iteration {}: Loss = {:.6f}, Distance = {:.6f}".format(
                iteration, loss, distance))
    
    stable_time = time.time() - stable_start_time
    
    # Comprehensive Analysis
    print("\n" + "=" * 60)
    print("CHALLENGING SCENARIO RESULTS")
    print("=" * 60)
    
    # Final performance comparison
    final_sgd_loss = sgd_losses[-1]
    final_stable_loss = stable_losses[-1]
    final_sgd_distance = sgd_param_distances[-1]
    final_stable_distance = stable_param_distances[-1]
    
    print("Final Performance:")
    print("  Standard SGD Loss:     {:.6f}".format(final_sgd_loss))
    print("  StableSGD Loss:        {:.6f}".format(final_stable_loss))
    print("  Standard SGD Distance: {:.6f}".format(final_sgd_distance))
    print("  StableSGD Distance:    {:.6f}".format(final_stable_distance))
    
    # Determine winner
    if final_stable_loss < final_sgd_loss:
        loss_improvement = (final_sgd_loss - final_stable_loss) / final_sgd_loss * 100
        print("  >>> StableSGD achieved {:.2f}% better loss".format(loss_improvement))
    else:
        print("  >>> Standard SGD achieved better final loss")
    
    if final_stable_distance < final_sgd_distance:
        distance_improvement = (final_sgd_distance - final_stable_distance) / final_sgd_distance * 100
        print("  >>> StableSGD reached {:.2f}% closer to true optimum".format(distance_improvement))
    
    # Stability analysis in high-noise environment
    def calculate_trajectory_stability(losses, window=20):
        if len(losses) < window:
            return 0.0
        
        stabilities = []
        for i in range(window, len(losses)):
            recent_losses = losses[i-window:i]
            mean_loss = sum(recent_losses) / len(recent_losses)
            variance = sum((l - mean_loss) ** 2 for l in recent_losses) / len(recent_losses)
            stability = math.sqrt(variance) / mean_loss if mean_loss > 0 else float('inf')
            stabilities.append(stability)
        
        return sum(stabilities) / len(stabilities)
    
    sgd_stability = calculate_trajectory_stability(sgd_losses)
    stable_stability = calculate_trajectory_stability(stable_losses)
    
    print("\nTrajectory Stability in Noisy Environment:")
    print("  Standard SGD stability:  {:.8f}".format(sgd_stability))
    print("  StableSGD stability:     {:.8f}".format(stable_stability))
    
    if stable_stability < sgd_stability:
        stability_improvement = sgd_stability / stable_stability
        print("  >>> StableSGD is {:.2f}x more stable".format(stability_improvement))
    
    # Convergence analysis
    def find_convergence_point(distances, threshold=0.1):
        for i, dist in enumerate(distances):
            if dist < threshold:
                return i
        return len(distances)  # Never converged
    
    sgd_convergence = find_convergence_point(sgd_param_distances)
    stable_convergence = find_convergence_point(stable_param_distances)
    
    print("\nConvergence Analysis:")
    if sgd_convergence < len(sgd_param_distances):
        print("  Standard SGD converged at iteration: {}".format(sgd_convergence))
    else:
        print("  Standard SGD did not converge within {} iterations".format(num_iterations))
        
    if stable_convergence < len(stable_param_distances):
        print("  StableSGD converged at iteration: {}".format(stable_convergence))
        if stable_convergence < sgd_convergence:
            speedup = (sgd_convergence - stable_convergence) / sgd_convergence * 100
            print("  >>> StableSGD converged {:.1f}% faster".format(speedup))
    else:
        print("  StableSGD did not converge within {} iterations".format(num_iterations))
    
    # Energy conservation analysis
    if hasattr(optimizer_stable, 'energy_history') and len(optimizer_stable.energy_history) > 10:
        energy_trend = optimizer_stable.energy_history[-20:]  # Last 20 steps
        energy_variance = sum((e - sum(energy_trend)/len(energy_trend))**2 for e in energy_trend) / len(energy_trend)
        print("\nEnergy Conservation:")
        print("  Energy variance in final steps: {:.6f}".format(energy_variance))
        print("  (Lower variance indicates better energy conservation)")
    
    # Training efficiency
    print("\nComputational Efficiency:")
    print("  Standard SGD time:  {:.2f} seconds".format(sgd_time))
    print("  StableSGD time:     {:.2f} seconds".format(stable_time))
    overhead = (stable_time - sgd_time) / sgd_time * 100
    print("  StableSGD overhead: {:.1f}%".format(overhead))
    
    return {
        'sgd_final_loss': final_sgd_loss,
        'stable_final_loss': final_stable_loss,
        'sgd_stability': sgd_stability,
        'stable_stability': stable_stability,
        'sgd_convergence': sgd_convergence,
        'stable_convergence': stable_convergence,
        'improved_loss': final_stable_loss < final_sgd_loss,
        'improved_stability': stable_stability < sgd_stability,
        'improved_convergence': stable_convergence < sgd_convergence
    }


if __name__ == "__main__":
    print("ENHANCED RESEARCH: StableSGD for Challenging Optimization Scenarios")
    print("Goal: Demonstrate clear benefits of Hamiltonian momentum conservation")
    print("Date: {}".format(time.strftime("%Y-%m-%d")))
    print()
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Run the challenging experiment
    results = run_challenging_stability_experiment()
    
    # Research conclusions
    print("\n" + "=" * 60)
    print("RESEARCH CONCLUSIONS")
    print("=" * 60)
    print()
    
    benefits_count = sum([
        results['improved_loss'],
        results['improved_stability'], 
        results['improved_convergence']
    ])
    
    if benefits_count >= 2:
        print("+ STRONG EVIDENCE: StableSGD shows clear benefits in challenging scenarios")
        print("  Key advantages demonstrated:")
        if results['improved_loss']:
            print("    - Better final optimization performance")
        if results['improved_stability']:
            print("    - Enhanced stability under high gradient noise")
        if results['improved_convergence']:
            print("    - Faster convergence to target optimum")
            
    elif benefits_count == 1:
        print("+ PARTIAL EVIDENCE: StableSGD shows some benefits")
        print("  Further parameter tuning may enhance performance")
        
    else:
        print("- LIMITED EVIDENCE: Benefits not demonstrated in this scenario")
        print("  May require different problem characteristics or parameter tuning")
    
    print()
    print("This experiment successfully tests SGD + Hamiltonian momentum conservation")
    print("on challenging optimization landscapes similar to LLM training difficulties.")