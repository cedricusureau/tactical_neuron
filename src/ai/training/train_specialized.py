# src/ai/training/train_specialized.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
import time

from .duel_environment import DuelEnvironment
from ..models.class_neural_controller import ClassCharacterAI
from src.ai.utils.state_encoder import default_encoder


def train_specialized_agents(episodes=1000, max_steps=100, batch_size=64,
                             gamma=0.95, epsilon_start=1.0, epsilon_end=0.1,
                             epsilon_decay=0.995, save_interval=100):
    """
    Fonction simplifiée pour entraîner des agents spécialisés qui s'affrontent
    """
    # Créer l'environnement
    env = DuelEnvironment()
    print("Environnement d'entraînement initialisé")

    # Réinitialiser pour configurer
    initial_state = env.reset()

    # Taille de l'état et de l'espace d'action
    state_size = default_encoder.state_size
    warrior_action_size = env.get_action_space_size("warrior")
    mage_action_size = env.get_action_space_size("mage")

    print(f"Taille d'état: {state_size}")
    print(f"Actions guerrier: {warrior_action_size}, Actions mage: {mage_action_size}")

    # Créer les agents spécialisés
    warrior_agent = ClassCharacterAI(state_size, 128, warrior_action_size, "warrior")
    mage_agent = ClassCharacterAI(state_size, 128, mage_action_size, "mage")

    # Créer les réseaux cibles (pour stabiliser l'apprentissage)
    warrior_target = ClassCharacterAI(state_size, 128, warrior_action_size, "warrior")
    warrior_target.model.load_state_dict(warrior_agent.model.state_dict())
    warrior_target.model.eval()

    mage_target = ClassCharacterAI(state_size, 128, mage_action_size, "mage")
    mage_target.model.load_state_dict(mage_agent.model.state_dict())
    mage_target.model.eval()

    # Mémoires d'expérience
    warrior_memory = []
    mage_memory = []

    # Métriques
    warrior_rewards = []
    mage_rewards = []
    episode_lengths = []
    warrior_wins = 0
    mage_wins = 0
    draws = 0

    # Paramètres d'exploration
    warrior_epsilon = epsilon_start
    mage_epsilon = epsilon_start

    # Dossier pour sauvegarder les modèles
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("data", "models", f"specialized_duel_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Début de l'entraînement - {episodes} épisodes")

    # Boucle d'entraînement
    for episode in tqdm(range(episodes)):
        # Réinitialiser l'environnement
        state = env.reset()
        warrior_state = state["warrior"]
        mage_state = state["mage"]

        warrior_episode_reward = 0
        mage_episode_reward = 0
        done = False

        # Boucle d'un épisode
        while not done and env.current_step < max_steps:
            # Déterminer qui joue ce tour
            current_character = env.turn_system.get_current_character()
            current_type = "warrior" if current_character == env.warrior else "mage"

            # Actions pour les deux joueurs
            actions = {}

            # Sélectionner l'action du joueur actif
            if current_type == "warrior":
                if np.random.random() < warrior_epsilon:
                    # Exploration
                    actions["warrior"] = np.random.randint(0, warrior_action_size)
                else:
                    # Exploitation
                    actions["warrior"] = warrior_agent.select_action(
                        torch.FloatTensor(warrior_state), is_training=True)
            else:  # mage
                if np.random.random() < mage_epsilon:
                    # Exploration
                    actions["mage"] = np.random.randint(0, mage_action_size)
                else:
                    # Exploitation
                    actions["mage"] = mage_agent.select_action(
                        torch.FloatTensor(mage_state), is_training=True)

            # Exécuter l'action
            next_state, rewards, done = env.step(actions)
            warrior_next_state = next_state["warrior"]
            mage_next_state = next_state["mage"]

            # Accumuler les récompenses
            warrior_reward = rewards.get("warrior", 0)
            mage_reward = rewards.get("mage", 0)
            warrior_episode_reward += warrior_reward
            mage_episode_reward += mage_reward

            # Stocker l'expérience
            if current_type == "warrior":
                warrior_memory.append(
                    (warrior_state, actions["warrior"], warrior_reward, warrior_next_state, done))
            else:  # mage
                mage_memory.append(
                    (mage_state, actions["mage"], mage_reward, mage_next_state, done))

            # Limiter la taille des mémoires
            if len(warrior_memory) > 10000:
                warrior_memory = warrior_memory[-10000:]
            if len(mage_memory) > 10000:
                mage_memory = mage_memory[-10000:]

            # Mettre à jour les états
            warrior_state = warrior_next_state
            mage_state = mage_next_state

        # Entraîner les agents si assez d'exemples
        if len(warrior_memory) >= batch_size:
            train_agent(warrior_agent, warrior_target, warrior_memory, batch_size, gamma)

        if len(mage_memory) >= batch_size:
            train_agent(mage_agent, mage_target, mage_memory, batch_size, gamma)

        # Mettre à jour les réseaux cibles périodiquement
        if episode % 10 == 0:
            warrior_target.model.load_state_dict(warrior_agent.model.state_dict())
            mage_target.model.load_state_dict(mage_agent.model.state_dict())

        # Décroissance epsilon
        warrior_epsilon = max(epsilon_end, warrior_epsilon * epsilon_decay)
        mage_epsilon = max(epsilon_end, mage_epsilon * epsilon_decay)

        # Enregistrer les métriques
        warrior_rewards.append(warrior_episode_reward)
        mage_rewards.append(mage_episode_reward)
        episode_lengths.append(env.current_step)

        # Déterminer le vainqueur
        if env.warrior.is_dead() and not env.mage.is_dead():
            mage_wins += 1
        elif env.mage.is_dead() and not env.warrior.is_dead():
            warrior_wins += 1
        elif env.mage.is_dead() and env.warrior.is_dead() or env.current_step >= max_steps:
            draws += 1

        # Sauvegarder périodiquement les modèles
        if (episode + 1) % save_interval == 0 or episode == episodes - 1:
            # Sauvegarder les modèles
            warrior_path = os.path.join(model_dir, f"warrior_model_ep{episode + 1}.pt")
            mage_path = os.path.join(model_dir, f"mage_model_ep{episode + 1}.pt")

            warrior_agent.save_model(warrior_path)
            mage_agent.save_model(mage_path)

            # Afficher les statistiques
            print(f"\nÉpisode {episode + 1}/{episodes}")
            print(f"Récompense moyenne guerrier (10 derniers): {np.mean(warrior_rewards[-10:]):.2f}")
            print(f"Récompense moyenne mage (10 derniers): {np.mean(mage_rewards[-10:]):.2f}")
            print(f"Longueur moyenne d'épisode (10 derniers): {np.mean(episode_lengths[-10:]):.2f}")
            print(f"Victoires: Guerrier={warrior_wins}, Mage={mage_wins}, Nul={draws}")
            print(f"Epsilon: Guerrier={warrior_epsilon:.3f}, Mage={mage_epsilon:.3f}")

    # Sauvegarder les modèles finaux
    warrior_final_path = os.path.join(model_dir, "warrior_model_final.pt")
    mage_final_path = os.path.join(model_dir, "mage_model_final.pt")

    warrior_agent.save_model(warrior_final_path)
    mage_agent.save_model(mage_final_path)

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(warrior_rewards, label='Guerrier')
    plt.plot(mage_rewards, label='Mage')
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths)
    plt.title('Longueur des épisodes')
    plt.xlabel('Épisode')
    plt.ylabel('Nombre d\'étapes')

    plt.subplot(1, 3, 3)
    win_count = [warrior_wins, mage_wins, draws]
    plt.bar(['Guerrier', 'Mage', 'Nul'], win_count)
    plt.title('Résultats des duels')
    plt.ylabel('Nombre de victoires')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "learning_curves.png"))

    print(f"\nEntraînement terminé. Modèles sauvegardés:")
    print(f"Guerrier: {warrior_final_path}")
    print(f"Mage: {mage_final_path}")

    return warrior_agent, mage_agent, warrior_final_path, mage_final_path


def train_agent(agent, target_network, memory, batch_size, gamma):
    """
    Entraîne un agent avec un mini-batch d'expériences
    """
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
        next_actions = agent.model(next_states).argmax(1, keepdim=True)
        target_q = target_network.model(next_states).gather(1, next_actions)
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


def evaluate_duel(warrior_model_path, mage_model_path, num_episodes=10, render=True, delay=0.5):
    """
    Évalue les agents spécialisés entraînés en les faisant s'affronter
    """
    # Créer l'environnement
    env = DuelEnvironment()

    # Réinitialiser pour obtenir la taille des états et actions
    initial_state = env.reset()

    # Taille de l'état et de l'espace d'action
    state_size = default_encoder.state_size
    warrior_action_size = env.get_action_space_size("warrior")
    mage_action_size = env.get_action_space_size("mage")

    # Charger les agents spécialisés
    warrior_agent = ClassCharacterAI(state_size, 128, warrior_action_size, "warrior")
    mage_agent = ClassCharacterAI(state_size, 128, mage_action_size, "mage")

    warrior_agent.load_model(warrior_model_path)
    mage_agent.load_model(mage_model_path)

    # Mettre les modèles en mode évaluation
    warrior_agent.model.eval()
    mage_agent.model.eval()

    # Statistiques
    warrior_wins = 0
    mage_wins = 0
    draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        warrior_state = state["warrior"]
        mage_state = state["mage"]

        done = False
        steps = 0

        print(f"\nÉpisode {episode + 1}/{num_episodes}")

        while not done:
            steps += 1

            # Déterminer qui joue ce tour
            current_character = env.turn_system.get_current_character()
            current_type = "warrior" if current_character == env.warrior else "mage"

            # Préparer les actions
            actions = {}

            # Action du guerrier
            if current_type == "warrior":
                actions["warrior"] = warrior_agent.select_action(torch.FloatTensor(warrior_state))

            # Action du mage
            elif current_type == "mage":
                actions["mage"] = mage_agent.select_action(torch.FloatTensor(mage_state))

            # Afficher l'action
            action_names = ["haut", "droite", "bas", "gauche", "attendre"]
            spell_names = ["sort 1", "sort 2", "sort 3", "sort 4", "sort 5"]

            action = actions.get(current_type, 0)
            if action < 5:
                action_name = action_names[action]
            else:
                spell_index = action - 5
                action_name = spell_names[spell_index] if spell_index < len(spell_names) else f"sort {spell_index + 1}"

            print(f"Tour {steps}: {current_type.capitalize()} choisit {action_name}")

            # Exécuter l'action
            next_state, rewards, done = env.step(actions)
            warrior_next_state = next_state["warrior"]
            mage_next_state = next_state["mage"]

            # Afficher l'état
            if render:
                env.render(mode='console')
                if delay > 0:
                    time.sleep(delay)  # Pause pour mieux voir l'évolution

            # Mettre à jour les états
            warrior_state = warrior_next_state
            mage_state = mage_next_state

        # Fin de l'épisode
        episode_lengths.append(steps)

        # Déterminer le vainqueur
        if env.warrior.is_dead() and not env.mage.is_dead():
            winner = "Mage"
            mage_wins += 1
        elif env.mage.is_dead() and not env.warrior.is_dead():
            winner = "Guerrier"
            warrior_wins += 1
        else:
            winner = "Match nul"
            draws += 1

        print(f"Fin de l'épisode {episode + 1} en {steps} tours. Vainqueur: {winner}")

    # Afficher les statistiques finales
    print("\n=== Résultats de l'évaluation ===")
    print(f"Épisodes: {num_episodes}")
    print(f"Victoires Guerrier: {warrior_wins} ({warrior_wins / num_episodes * 100:.1f}%)")
    print(f"Victoires Mage: {mage_wins} ({mage_wins / num_episodes * 100:.1f}%)")
    print(f"Matchs nuls: {draws} ({draws / num_episodes * 100:.1f}%)")
    print(f"Durée moyenne des combats: {np.mean(episode_lengths):.1f} tours")

    return warrior_wins, mage_wins, draws


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement et évaluation d'agents spécialisés")
    parser.add_argument("--train", action="store_true", help="Entraîner les agents")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer les agents")
    parser.add_argument("--warrior-model", type=str, help="Chemin vers le modèle du guerrier")
    parser.add_argument("--mage-model", type=str, help="Chemin vers le modèle du mage")
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--no-render", action="store_true", help="Désactiver l'affichage lors de l'évaluation")
    parser.add_argument("--delay", type=float, default=0.5, help="Délai entre les actions lors de l'évaluation")

    args = parser.parse_args()

    if args.train:
        warrior_agent, mage_agent, warrior_path, mage_path = train_specialized_agents(episodes=args.episodes)

        # Si entraînement suivi d'évaluation
        if args.evaluate:
            evaluate_duel(
                warrior_model_path=warrior_path,
                mage_model_path=mage_path,
                num_episodes=args.eval_episodes,
                render=not args.no_render,
                delay=args.delay
            )
    elif args.evaluate and args.warrior_model and args.mage_model:
        evaluate_duel(
            warrior_model_path=args.warrior_model,
            mage_model_path=args.mage_model,
            num_episodes=args.eval_episodes,
            render=not args.no_render,
            delay=args.delay
        )
    else:
        parser.print_help()