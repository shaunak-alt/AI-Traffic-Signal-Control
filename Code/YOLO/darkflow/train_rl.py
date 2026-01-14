"""
Training Script for SAC+GNN Traffic Signal Control.

Usage:
    python train_rl.py                      # Train with default settings
    python train_rl.py --episodes 100       # Train for 100 episodes
    python train_rl.py --render             # Train with visualization
    python train_rl.py --eval               # Evaluate trained model
    python train_rl.py --headless           # Train without any visualization
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Optional tensorboard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from traffic_env import TrafficEnv
from sac_gnn_agent import SACGNNAgent


def train(
    agent: SACGNNAgent,
    env: TrafficEnv,
    num_episodes: int = 100,
    max_steps_per_episode: int = 300,
    updates_per_step: int = 1,
    eval_interval: int = 10,
    save_interval: int = 20,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True
) -> Dict[str, List]:
    """
    Train the SAC+GNN agent.
    
    Args:
        agent: SAC agent to train
        env: Traffic environment
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        updates_per_step: Number of gradient updates per environment step
        eval_interval: Evaluate every N episodes
        save_interval: Save checkpoint every N episodes
        log_dir: Directory for tensorboard logs
        checkpoint_dir: Directory for model checkpoints
        verbose: Print training progress
    
    Returns:
        Dictionary containing training history
    """
    # Setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = None
    if TENSORBOARD_AVAILABLE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(os.path.join(log_dir, f"sac_gnn_{timestamp}"))
    
    # Training history
    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "vehicles_crossed": [],
        "avg_waiting": [],
        "critic_loss": [],
        "actor_loss": [],
        "alpha": []
    }
    
    total_steps = 0
    best_reward = float('-inf')
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_crossed = 0
        episode_waiting = []
        
        start_time = time.time()
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Update agent
            for _ in range(updates_per_step):
                stats = agent.update()
                if stats:
                    history["critic_loss"].append(stats.get("critic_loss", 0))
                    history["actor_loss"].append(stats.get("actor_loss", 0))
                    history["alpha"].append(stats.get("alpha", 0))
            
            # Track metrics
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            episode_crossed = info.get("total_crossed", 0)
            episode_waiting.append(info.get("waiting_vehicles", 0))
            
            obs = next_obs
            
            if done:
                break
        
        # Episode complete
        episode_time = time.time() - start_time
        avg_waiting = np.mean(episode_waiting) if episode_waiting else 0
        
        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(episode_steps)
        history["vehicles_crossed"].append(episode_crossed)
        history["avg_waiting"].append(avg_waiting)
        
        # Logging
        if writer:
            writer.add_scalar("train/episode_reward", episode_reward, episode)
            writer.add_scalar("train/episode_length", episode_steps, episode)
            writer.add_scalar("train/vehicles_crossed", episode_crossed, episode)
            writer.add_scalar("train/avg_waiting", avg_waiting, episode)
            if stats:
                writer.add_scalar("train/critic_loss", stats.get("critic_loss", 0), episode)
                writer.add_scalar("train/actor_loss", stats.get("actor_loss", 0), episode)
                writer.add_scalar("train/alpha", stats.get("alpha", 0), episode)
        
        if verbose:
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Crossed: {episode_crossed:4d} | "
                  f"Waiting: {avg_waiting:5.1f} | "
                  f"Steps: {episode_steps:4d} | "
                  f"Time: {episode_time:.1f}s")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"))
        
        # Periodic save
        if episode % save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt"))
        
        # Evaluation
        if episode % eval_interval == 0:
            eval_reward = evaluate(agent, env, num_episodes=3, verbose=False)
            if writer:
                writer.add_scalar("eval/episode_reward", eval_reward, episode)
            if verbose:
                print(f"  --> Eval reward: {eval_reward:.2f}")
    
    # Final save
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))
    
    if writer:
        writer.close()
    
    return history


def evaluate(
    agent: SACGNNAgent,
    env: TrafficEnv,
    num_episodes: int = 5,
    max_steps: int = 300,
    verbose: bool = True
) -> float:
    """
    Evaluate the trained agent.
    
    Args:
        agent: Trained SAC agent
        env: Traffic environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Print evaluation progress
    
    Returns:
        Average episode reward
    """
    total_rewards = []
    total_crossed = []
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_crossed.append(info.get("total_crossed", 0))
        
        if verbose:
            print(f"Eval Episode {episode} | Reward: {episode_reward:.2f} | Crossed: {info.get('total_crossed', 0)}")
    
    avg_reward = np.mean(total_rewards)
    avg_crossed = np.mean(total_crossed)
    
    if verbose:
        print(f"\nAverage Reward: {avg_reward:.2f}")
        print(f"Average Vehicles Crossed: {avg_crossed:.1f}")
    
    return avg_reward


def main():
    parser = argparse.ArgumentParser(description="Train SAC+GNN for Traffic Signal Control")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode (default 60 for faster training)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Network parameters (smaller for faster CPU training)
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden dimension")
    parser.add_argument("--gnn-hidden", type=int, default=32, help="GNN hidden dimension")
    parser.add_argument("--gnn-output", type=int, default=32, help="GNN output dimension")
    
    # Environment parameters
    parser.add_argument("--sim-steps", type=int, default=5, help="Simulation steps per action")
    parser.add_argument("--min-green", type=int, default=10, help="Minimum green time")
    parser.add_argument("--max-green", type=int, default=60, help="Maximum green time")
    
    # Mode
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--headless", action="store_true", help="No rendering at all")
    parser.add_argument("--eval", action="store_true", help="Evaluate only (requires checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for eval/resume")
    
    # Directories
    parser.add_argument("--log-dir", type=str, default="logs", help="Tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Setup render mode
    if args.headless:
        render_mode = None
    elif args.render:
        render_mode = "human"
    else:
        render_mode = None
    
    # Create environment
    env = TrafficEnv(
        render_mode=render_mode,
        sim_steps_per_action=args.sim_steps,
        max_episode_steps=args.max_steps,
        min_green=args.min_green,
        max_green=args.max_green
    )
    
    # Create agent
    agent = SACGNNAgent(
        obs_dim=16,
        action_dim=4,
        hidden_dim=args.hidden_dim,
        gnn_hidden_dim=args.gnn_hidden,
        gnn_output_dim=args.gnn_output,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size
    )
    
    print(f"Device: {agent.device}")
    print(f"Render mode: {render_mode}")
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    
    if args.eval:
        # Evaluation mode
        if not args.checkpoint:
            # Try to load best model
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            if os.path.exists(best_path):
                agent.load(best_path)
                print(f"Loaded best model from {best_path}")
            else:
                print("Warning: No checkpoint found, using untrained model")
        
        evaluate(agent, env, num_episodes=10, verbose=True)
    else:
        # Training mode
        print(f"\n{'='*60}")
        print("Starting SAC+GNN Training for Traffic Signal Control")
        print(f"{'='*60}")
        print(f"Episodes: {args.episodes}")
        print(f"Hidden dim: {args.hidden_dim}, GNN hidden: {args.gnn_hidden}")
        print(f"{'='*60}\n")
        
        history = train(
            agent=agent,
            env=env,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir
        )
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")
        if TENSORBOARD_AVAILABLE:
            print(f"Tensorboard logs: {args.log_dir}")
            print(f"Run: tensorboard --logdir {args.log_dir}")
        print(f"{'='*60}")
    
    env.close()


if __name__ == "__main__":
    main()
