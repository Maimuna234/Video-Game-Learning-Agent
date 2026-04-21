import streamlit as st
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
import os

# 1. Re-define the Q-Network architecture exactly as it is in your notebook
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 2. Set up the Streamlit UI
st.title("🚀 Lunar Lander RL Agent")
st.write("This is a Deep Q-Network (DQN) trained to safely land the module.")

# Load the model weights
@st.cache_resource
def load_model():
    model = QNetwork(state_size=8, action_size=4)
    # Use map_location='cpu' ensuring it works on Streamlit's CPU-only servers
    model.load_state_dict(torch.load('lunar_lander_model.pth', map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode
    return model

model = load_model()

# 3. Create a button to run the agent
if st.button("Run Agent and Generate Video"):
    with st.spinner("Agent is flying... capturing frames..."):
        # Initialize environment to render RGB arrays for our GIF
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        state, _ = env.reset()
        done = False
        
        frames = []
        score = 0
        
        while not done:
            # Capture the current frame
            frames.append(env.render())
            
            # Convert state to tensor and get action (Exploitation: no epsilon)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = model(state_tensor)
            action = np.argmax(action_values.numpy())
            
            # Take step in environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            
        env.close()
        
        # 4. Save frames as a GIF and display it
        gif_path = "lander_demo.gif"
        imageio.mimsave(gif_path, frames, fps=30)
        
        st.success(f"Episode finished with a Total Reward of: {score:.2f}")
        st.image(gif_path, caption="Agent Gameplay")
