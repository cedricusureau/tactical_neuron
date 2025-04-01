# src/ai/training/train_rl.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse

from .rl_environment import RLEnvironment
from ..models.neural_controller import CharacterAI, NeuralNetwork


def train_agent(episodes=1000, max_steps=100, learning_rate=0.001, batch_size=64,
                gamma=0.95, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                target_update=10, memory_size=10000, board_width=10, board_height=10):
    """
    Entraîne un agent par reinforcement learning
    """
    # Créer l'environnement
    env = RLEnvironment(board_width, board_height, max_steps)

    # Obtenir la taille de l'état d'observation
    state = env.reset()
    state_size = len(state)
    action_size = 5  # 4 directions + ne rien faire

    # Créer l'agent
    agent = CharacterAI(state_size, 64, action_size)

    # Créer un réseau cible (pour stabiliser l'apprentissage)
    target_network = NeuralNetwork(state_size, 64, action_size)
    target_network.load_state_dict(agent.model.state_dict())
    target_network.eval()

    # Mémoire d'expérience (experience replay)
    memory = []

    # Métriques pour suivre les performances
    rewards_history = []
    steps_history = []
    epsilon = epsilon_start

    # Dossier pour sauvegarder les modèles
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("data", "models", f"training_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Début de l'entraînement - {episodes} épisodes")

    # Boucle d'entraînement
    for episode in tqdm(range(episodes)):
        # Réinitialiser l'environnement
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        # Boucle d'un épisode
        while not done and step < max_steps:
            # Sélectionner une action avec epsilon-greedy
            if np.random.random() < epsilon:
                # Exploration: action aléatoire
                action = np.random.randint(0, action_size)
            else:
                # Exploitation: meilleure action selon le modèle
                with torch.no_grad():
                    q_values = agent.model(torch.FloatTensor(state))
                    action = torch.argmax(q_values).item()

            # Exécuter l'action
            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1

            # Stocker l'expérience
            memory.append((state, action, reward, next_state, done))

            # Limiter la taille de la mémoire
            if len(memory) > memory_size:
                memory.pop(0)

            # Entraîner l'agent si assez d'exemples
            if len(memory) >= batch_size:
                # Échantillonner un mini-batch
                indices = np.random.choice(len(memory), batch_size, replace=False)
                batch = [memory[i] for i in indices]
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convertir en tenseurs
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Calculer les valeurs Q actuelles
                current_q = agent.model(states).gather(1, actions)

                # Calculer les valeurs Q cibles (sans gradient)
                with torch.no_grad():
                    # Double DQN: utiliser le réseau principal pour choisir l'action
                    # et le réseau cible pour estimer sa valeur
                    next_actions = agent.model(next_states).argmax(1, keepdim=True)
                    target_q = target_network(next_states).gather(1, next_actions)
                    target_q = rewards + gamma * target_q * (1 - dones)

                # Calculer la perte
                loss = nn.SmoothL1Loss()(current_q, target_q)

                # Mettre à jour les poids
                agent.optimizer.zero_grad()
                loss.backward()
                # Limiter la norme du gradient (pour stabilité)
                for param in agent.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                agent.optimizer.step()

            # Passer à l'état suivant
            state = next_state

        # Mettre à jour le réseau cible périodiquement
        if episode % target_update == 0:
            target_network.load_state_dict(agent.model.state_dict())

        # Décroissance epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Enregistrer les métriques
        rewards_history.append(total_reward)
        steps_history.append(step)

        # Sauvegarder périodiquement le modèle
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            model_path = os.path.join(model_dir, f"model_ep{episode + 1}.pt")
            agent.save_model(model_path)

            # Afficher les performances
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            print(f"Épisode {episode + 1}/{episodes} - Récompense moyenne (100 épisodes): {avg_reward:.2f}, "
                  f"Étapes moyennes: {avg_steps:.2f}, Epsilon: {epsilon:.3f}")

    # Sauvegarder le modèle final
    final_path = os.path.join(model_dir, "model_final.pt")
    agent.save_model(final_path)

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')

    plt.subplot(1, 2, 2)
    plt.plot(steps_history)
    plt.title('Étapes par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Nombre d\'étapes')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "learning_curve.png"))

    print(f"Entraînement terminé. Modèle sauvegardé: {final_path}")
    return agent, final_path


def evaluate_agent(model_path, num_episodes=10, render=True):
    """
    Évalue un agent entraîné
    """
    # Créer l'environnement
    env = RLEnvironment()

    # Charger l'agent
    state = env.reset()
    state_size = len(state)
    agent = CharacterAI(state_size, 64, 5)
    agent.load_model(model_path)
    agent.model.eval()

    # Suivi des performances
    rewards = []
    steps = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done:
            # Choisir l'action optimale
            with torch.no_grad():
                q_values = agent.model(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()

            # Exécuter l'action
            next_state, reward, done = env.step(action)
            total_reward += reward
            step += 1

            # Afficher l'état
            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                env.render(mode='console')
                print(f"Épisode {episode + 1}/{num_episodes} - Étape {step}")
                print(f"Action: {['haut', 'droite', 'bas', 'gauche', 'attendre'][action]}")
                print(f"Récompense: {reward:.2f}, Total: {total_reward:.2f}")
                input("Appuyez sur Entrée pour continuer...")

            state = next_state

        rewards.append(total_reward)
        steps.append(step)
        print(f"Épisode {episode + 1}/{num_episodes} - Récompense: {total_reward:.2f}, Étapes: {step}")

    print(f"Évaluation terminée. Récompense moyenne: {np.mean(rewards):.2f}, Étapes moyennes: {np.mean(steps):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement par RL d'un agent pour le jeu tactique")
    parser.add_argument("--train", action="store_true", help="Entraîner un agent")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer un agent")
    parser.add_argument("--model", type=str, help="Chemin vers un modèle à évaluer")
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Nombre d'épisodes d'évaluation")

    args = parser.parse_args()

    if args.train:
        train_agent(episodes=args.episodes)
    elif args.evaluate and args.model:
        evaluate_agent(args.model, num_episodes=args.eval_episodes)
    else:
        parser.print_help()