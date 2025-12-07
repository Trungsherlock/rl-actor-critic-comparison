# """
# Value Network (Critic) for Actor-Critic Algorithm - FIXED INITIALIZATION

# This network learns to estimate how good a state is.
# It's the "Critic" that judges the Actor's actions.

# FIXED: Proper initialization for environments with negative rewards (like MountainCar)
# """

# import torch
# import torch.nn as nn


# class ValueNetwork(nn.Module):
#     """
#     The Critic network that estimates state values.
    
#     This answers the question: "How good is this state?"
#     For example: "If I'm in this state, how much total reward will I get?"
#     """
    
#     def __init__(self, state_dim, hidden_size=128):
#         """
#         Initialize the Value Network.
        
#         Args:
#             state_dim: number of features in the state
#                       (4 for CartPole: [position, velocity, angle, angular_velocity])
#                       (2 for MountainCar: [position, velocity])
#             hidden_size: number of neurons in each hidden layer
#         """
#         super(ValueNetwork, self).__init__()
        
#         # Store dimensions for reference
#         self.state_dim = state_dim
#         self.hidden_size = hidden_size
        
#         self.layer1 = nn.Linear(state_dim, hidden_size)
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.layer3 = nn.Linear(hidden_size, 1)
        
#         self.activation = nn.ReLU()
        
#         # CRITICAL FIX: Initialize output bias to negative value
#         # For MountainCar, we expect returns around -1000 to -100
#         # Start with a reasonable negative estimate
#         self.layer3.bias.data.fill_(-100.0)
        
#         # Optional: Scale down output layer weights for stability
#         self.layer3.weight.data.mul_(0.1)
        
#     def forward(self, state):
#         """
#         Forward pass: compute the value of a state.
        
#         This is how we get from state -> value estimate.
        
#         Args:
#             state: current state as a tensor
#                   Shape: (batch_size, state_dim) or (state_dim,)
        
#         Returns:
#             value: estimated value of the state
#                   Shape: (batch_size, 1) or (1,)
        
#         Example:
#             state = [-0.5, 0.0]  # MountainCar state
#             value = critic(state)  # might output: -500
#             This means: "From this state, I expect to get ~-500 total reward"
#         """

#         hidden1 = self.layer1(state)
#         hidden1_activated = self.activation(hidden1)
#         hidden2 = self.layer2(hidden1_activated)
#         hidden2_activated = self.activation(hidden2)
        
#         value = self.layer3(hidden2_activated)
        
#         return value
    
#     def predict_value(self, state):
#         """
#         Convenience function to predict value from a numpy array.
        
#         This handles conversion from numpy -> tensor -> prediction.
#         Useful for debugging or evaluation.
        
#         Args:
#             state: state as numpy array
        
#         Returns:
#             value: predicted value as a float
#         """
#         # Convert numpy array to tensor
#         state_tensor = torch.FloatTensor(state)
        
#         # If state is 1D, add batch dimension
#         if len(state_tensor.shape) == 1:
#             state_tensor = state_tensor.unsqueeze(0)
        
#         # Get prediction (no gradients needed for inference)
#         with torch.no_grad():
#             value = self.forward(state_tensor)
        
#         # Return as Python float
#         return value.item()


# # Example usage and testing
# if __name__ == "__main__":
#     print("Testing Value Network with Negative Initialization")
#     print("="*60)
    
#     # Test: MountainCar (state_dim = 2)
#     print("\nTest: MountainCar")
#     print("-" * 40)
    
#     mountaincar_state_dim = 2
#     critic = ValueNetwork(state_dim=mountaincar_state_dim, hidden_size=128)
    
#     print(f"Created Value Network:")
#     print(f"  State dimension: {mountaincar_state_dim}")
#     print(f"  Hidden size: {128}")
#     print(f"  Total parameters: {sum(p.numel() for p in critic.parameters())}")
    
#     # Check initial bias
#     print(f"\nInitial output bias: {critic.layer3.bias.item():.2f}")
    
#     # Create a fake MountainCar state
#     fake_state = torch.FloatTensor([[-0.5, 0.02]])
#     print(f"\nInput state: {fake_state[0].tolist()}")
    
#     # Get value prediction
#     value = critic(fake_state)
#     print(f"Initial predicted value: {value.item():.2f}")
#     print("  (Should be around -100, not near 0!)")
    
#     # Test gradient flow
#     print("\n" + "="*60)
#     print("Test: Gradient Check")
#     print("-" * 40)
    
#     # Simulate one training step
#     test_state = torch.FloatTensor([[-0.5, 0.0]])
#     predicted_value = critic(test_state)
    
#     # Target: if we got -1 reward and next state value is -99
#     target_value = torch.FloatTensor([[-100.0]])  # -1 + 0.99*(-99) ≈ -100
#     loss = (predicted_value - target_value) ** 2
    
#     print(f"Predicted value: {predicted_value.item():.2f}")
#     print(f"Target value: {target_value.item():.2f}")
#     print(f"Loss: {loss.item():.4f}")
    
#     # Backward pass
#     loss.backward()
    
#     # Check gradients
#     has_gradients = any(p.grad is not None for p in critic.parameters())
#     print(f"Gradients computed: {has_gradients}")
    
#     if has_gradients:
#         print("✓ Network is ready for training with negative rewards!")
    
#     print("\n" + "="*60)
#     print("All tests passed! Value Network properly initialized.")
#     print("="*60)

"""
Value Network - PROPERLY INITIALIZED for MountainCar

Key insight: MountainCar episodes typically last 200-1000 steps
So initial value estimates should be around -200 to -1000, not -100
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Value network with proper initialization for MountainCar."""
    
    def __init__(self, state_dim, hidden_size=128):
        """Initialize the Value Network."""
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        
        self.activation = nn.ReLU()
        
        # CRITICAL: Initialize to expect long episodes with -1 rewards
        # MountainCar: typical episode = 200-1000 steps
        # Expected return = -200 to -1000
        # Start in middle: -500
        self.layer3.bias.data.fill_(-500.0)
        
        # Scale weights for stable initialization
        self.layer3.weight.data.mul_(0.01)
        
    def forward(self, state):
        """Forward pass: compute the value of a state."""
        hidden1 = self.layer1(state)
        hidden1_activated = self.activation(hidden1)
        hidden2 = self.layer2(hidden1_activated)
        hidden2_activated = self.activation(hidden2)
        
        value = self.layer3(hidden2_activated)
        
        return value