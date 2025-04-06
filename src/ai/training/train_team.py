# src/ai/training/train_team.py
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
import random

from .team_environment import TeamEnvironment
from ..models.neural_controller import CharacterAI  # Utilisation de l'architecture standard
from src.ai.utils.state_encoder import default_encoder


def train_team_agents(episodes=1000, max_steps=50, batch_size=64,
                      gamma=0.95, epsilon_start=1.0, epsilon_end=0.1,
                      epsilon_decay=0.995, save_interval=100):
    """
    Entraîne des agents pour combats en équipe (3v3)
    """
    # Créer l'environnement
    env = TeamEnvironment()
    print("Environnement d'entraînement en équipe initialisé")

    # Réinitialiser pour configurer
    initial_state = env.reset()

    # Taille de l'état
    state_size = default_encoder.state_size

    # Créer un agent par classe de personnage
    agents = {}
    target_networks = {}
    memory_buffers = {}

    for character_class in env.character_classes:
        # Obtenir la taille de l'espace d'action pour cette classe
        action_size = env.get_action_space_size(character_class)

        # Créer l'agent avec l'architecture standard
        agents[character_class] = CharacterAI(state_size, 128, action_size)

        # Créer le réseau cible
        target_networks[character_class] = CharacterAI(state_size, 128, action_size)
        target_networks[character_class].model.load_state_dict(agents[character_class].model.state_dict())
        target_networks[character_class].model.eval()

        # Mémoire d'expérience pour cette classe
        memory_buffers[character_class] = []

    # Métriques de suivi
    team0_wins = 0
    team1_wins = 0
    draws = 0
    rewards_history = {character_class: [] for character_class in env.character_classes}
    episode_lengths = []

    # Paramètres d'exploration
    epsilons = {character_class: epsilon_start for character_class in env.character_classes}

    # Dossier pour sauvegarder les modèles
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("data", "models", f"team_training_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Début de l'entraînement en équipe - {episodes} épisodes")

    # Boucle d'entraînement
    for episode in tqdm(range(episodes)):
        # Réinitialiser l'environnement (génère une nouvelle composition d'équipe aléatoire)
        states = env.reset()

        # Enregistrer la composition des équipes
        team0_classes = env.team0_comp
        team1_classes = env.team1_comp

        # Réinitialiser les récompenses pour cet épisode
        episode_rewards = {character_class: 0.0 for character_class in env.character_classes}
        episode_counts = {character_class: 0 for character_class in env.character_classes}

        done = False

        # Conserver toutes les expériences de l'épisode
        episode_memories = {character_class: [] for character_class in env.character_classes}

        # Boucle d'un épisode
        while not done and env.current_step < max_steps:
            # Déterminer qui joue ce tour
            current_character = env.turn_system.get_current_character()
            if not current_character:
                break

            current_class = current_character.character_class
            current_team = 0 if current_character in env.team0 else 1
            current_id = f"team{current_team}_{current_character.id}"

            # Obtenir l'état actuel du personnage
            current_state = states.get(current_id)
            if current_state is None:
                # Si le personnage n'a plus d'état (mort), passer au suivant
                env.turn_system.next_turn()
                continue

            # Sélectionner une action avec epsilon-greedy
            if np.random.random() < epsilons[current_class]:
                # Exploration: action aléatoire
                action = np.random.randint(0, env.get_action_space_size(current_class))
            else:
                # Exploitation: meilleure action selon le modèle
                action = agents[current_class].select_action(torch.FloatTensor(current_state))

            # Exécuter l'action
            next_states, rewards, done = env.step({current_id: action})

            # Obtenir le nouvel état et la récompense pour le personnage actif
            next_state = next_states.get(current_id)
            if next_state is None:
                # Si le personnage est mort après son action, créer un état nul
                next_state = torch.zeros_like(current_state)

            reward = rewards.get(current_id, 0.0)

            # Stocker l'expérience temporairement
            episode_memories[current_class].append((current_state, action, reward, next_state, done))

            # Accumuler les récompenses par classe
            episode_rewards[current_class] += reward
            episode_counts[current_class] += 1

            # Mettre à jour les états
            states = next_states

        # Fin de l'épisode - stocker les expériences avec les récompenses finales
        # Cela permet de propager la récompense finale (victoire/défaite) à toutes les actions
        for character_class in env.character_classes:
            for state, action, _, next_state, is_done in episode_memories[character_class]:
                # Récupérer la récompense finale pour ce type de personnage et son équipe
                is_team0 = any(
                    char.character_class == character_class and char in env.team0 for char in env.team0 + env.team1)
                final_reward = 0.0

                if done:  # Si l'épisode est terminé
                    team0_alive = any(character in env.board.characters for character in env.team0)
                    team1_alive = any(character in env.board.characters for character in env.team1)

                    if team0_alive and not team1_alive:  # Team 0 gagne
                        final_reward = 1.0 if is_team0 else -1.0
                    elif team1_alive and not team0_alive:  # Team 1 gagne
                        final_reward = -1.0 if is_team0 else 1.0
                    else:  # Match nul
                        final_reward = -0.5

                # Stocker l'expérience avec la récompense finale
                memory_buffers[character_class].append((state, action, final_reward, next_state, is_done))

                # Limiter la taille des mémoires
                if len(memory_buffers[character_class]) > 100000:  # Buffer plus grand pour training par équipe
                    memory_buffers[character_class] = memory_buffers[character_class][-100000:]

        # Entraîner les agents si assez d'exemples
        for character_class in env.character_classes:
            # Calculer la récompense moyenne pour cette classe dans cet épisode
            avg_reward = episode_rewards[character_class] / max(1, episode_counts[character_class])
            rewards_history[character_class].append(avg_reward)

            # Entraîner l'agent si assez d'exemples
            if len(memory_buffers[character_class]) >= batch_size:
                for _ in range(4):  # Plusieurs updates par épisode pour accélérer l'apprentissage
                    train_agent(agents[character_class], target_networks[character_class],
                                memory_buffers[character_class], batch_size, gamma)

            # Mettre à jour le réseau cible périodiquement
            if episode % 10 == 0:
                target_networks[character_class].model.load_state_dict(agents[character_class].model.state_dict())

            # Décroissance epsilon
            epsilons[character_class] = max(epsilon_end, epsilons[character_class] * epsilon_decay)

        # Enregistrer la longueur de l'épisode
        episode_lengths.append(env.current_step)

        # Déterminer le vainqueur
        team0_alive = any(character in env.board.characters for character in env.team0)
        team1_alive = any(character in env.board.characters for character in env.team1)

        if team0_alive and not team1_alive:
            team0_wins += 1
        elif team1_alive and not team0_alive:
            team1_wins += 1
        else:
            draws += 1

        # Sauvegarder périodiquement les modèles
        if (episode + 1) % save_interval == 0 or episode == episodes - 1:
            for character_class in env.character_classes:
                model_path = os.path.join(model_dir, f"{character_class}_model_ep{episode + 1}.pt")
                agents[character_class].save_model(model_path)

            # Afficher les statistiques
            print(f"\nÉpisode {episode + 1}/{episodes}")
            for character_class in env.character_classes:
                print(
                    f"Récompense moyenne {character_class} (10 derniers): {np.mean(rewards_history[character_class][-10:]):.2f}")

            print(f"Longueur moyenne d'épisode (10 derniers): {np.mean(episode_lengths[-10:]):.2f}")
            print(f"Victoires: Équipe0={team0_wins}, Équipe1={team1_wins}, Nul={draws}")
            print(f"Epsilon: {epsilons}")

    # Sauvegarder les modèles finaux
    final_paths = {}
    for character_class in env.character_classes:
        final_path = os.path.join(model_dir, f"{character_class}_model_final.pt")
        agents[character_class].save_model(final_path)
        final_paths[character_class] = final_path

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for character_class in env.character_classes:
        plt.plot(rewards_history[character_class], label=character_class)
    plt.title('Récompenses par classe')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense moyenne')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths)
    plt.title('Longueur des épisodes')
    plt.xlabel('Épisode')
    plt.ylabel('Nombre d\'étapes')

    plt.subplot(1, 3, 3)
    win_count = [team0_wins, team1_wins, draws]
    plt.bar(['Équipe 0', 'Équipe 1', 'Nul'], win_count)
    plt.title('Résultats des combats')
    plt.ylabel('Nombre de victoires')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "learning_curves.png"))

    print(f"\nEntraînement terminé. Modèles sauvegardés dans {model_dir}")

    return agents, final_paths


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


def evaluate_team(warrior_model=None, mage_model=None, archer_model=None, num_episodes=5, render=True, delay=0.5):
    """
    Évalue les agents en mode équipe
    """
    # Créer l'environnement
    env = TeamEnvironment()

    # Charger les modèles s'ils sont fournis
    models = {}
    if warrior_model:
        models["warrior"] = warrior_model
    if mage_model:
        models["mage"] = mage_model
    if archer_model:
        models["archer"] = archer_model

    # Créer et charger les agents
    agents = {}
    state_size = default_encoder.state_size

    for character_class in env.character_classes:
        action_size = env.get_action_space_size(character_class)
        agents[character_class] = CharacterAI(state_size, 128, action_size)

        if character_class in models and models[character_class]:
            try:
                agents[character_class].load_model(models[character_class])
                print(f"Modèle pour {character_class} chargé: {models[character_class]}")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle pour {character_class}: {e}")

    # Statistiques
    team0_wins = 0
    team1_wins = 0
    draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        states = env.reset()
        done = False

        print(f"\n=== Épisode {episode + 1}/{num_episodes} ===")
        print(f"Équipe 0: {env.team0_comp}")
        print(f"Équipe 1: {env.team1_comp}")

        if render:
            env.render()
            if delay > 0:
                time.sleep(delay)

        while not done:
            # Déterminer qui joue ce tour
            current_character = env.turn_system.get_current_character()
            if not current_character:
                break

            current_class = current_character.character_class
            current_team = 0 if current_character in env.team0 else 1
            current_id = f"team{current_team}_{current_character.id}"

            # Obtenir l'état actuel du personnage
            current_state = states.get(current_id)
            if current_state is None:
                # Si le personnage n'a plus d'état (mort), passer au suivant
                env.turn_system.next_turn()
                continue

            # Sélectionner une action avec le modèle
            action = agents[current_class].select_action(torch.FloatTensor(current_state))

            # Exécuter l'action
            states, _, done = env.step({current_id: action})

            if render:
                env.render()
                if delay > 0:
                    time.sleep(delay)

        # Déterminer le vainqueur
        team0_alive = any(character in env.board.characters for character in env.team0)
        team1_alive = any(character in env.board.characters for character in env.team1)

        episode_lengths.append(env.current_step)

        if team0_alive and not team1_alive:
            team0_wins += 1
            print("Équipe 0 gagne!")
        elif team1_alive and not team0_alive:
            team1_wins += 1
            print("Équipe 1 gagne!")
        else:
            draws += 1
            print("Match nul!")

    # Afficher les statistiques finales
    print("\n=== Résultats de l'évaluation ===")
    print(f"Épisodes: {num_episodes}")
    print(f"Victoires Équipe 0: {team0_wins} ({team0_wins / num_episodes * 100:.1f}%)")
    print(f"Victoires Équipe 1: {team1_wins} ({team1_wins / num_episodes * 100:.1f}%)")
    print(f"Matchs nuls: {draws} ({draws / num_episodes * 100:.1f}%)")
    print(f"Durée moyenne des combats: {np.mean(episode_lengths):.1f} tours")

    return team0_wins, team1_wins, draws


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement et évaluation d'agents en équipe")
    parser.add_argument("--train", action="store_true", help="Entraîner les agents")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer les agents")
    parser.add_argument("--warrior-model", type=str, help="Chemin vers le modèle du guerrier")
    parser.add_argument("--mage-model", type=str, help="Chemin vers le modèle du mage")
    parser.add_argument("--archer-model", type=str, help="Chemin vers le modèle de l'archer")
    parser.add_argument("--episodes", type=int, default=1000, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--no-render", action="store_true", help="Désactiver l'affichage lors de l'évaluation")
    parser.add_argument("--delay", type=float, default=0.5, help="Délai entre les actions lors de l'évaluation")
    parser.add_argument("--save-interval", type=int, default=100, help="Intervalle pour sauvegarder les modèles")

    args = parser.parse_args()

    if args.train:
        agents, model_paths = train_team_agents(episodes=args.episodes, save_interval=args.save_interval)

        # Si entraînement suivi d'évaluation
        if args.evaluate:
            evaluate_team(
                warrior_model=model_paths.get("warrior"),
                mage_model=model_paths.get("mage"),
                archer_model=model_paths.get("archer"),
                num_episodes=args.eval_episodes,
                render=not args.no_render,
                delay=args.delay
            )
    elif args.evaluate:
        evaluate_team(
            warrior_model=args.warrior_model,
            mage_model=args.mage_model,
            archer_model=args.archer_model,
            num_episodes=args.eval_episodes,
            render=not args.no_render,
            delay=args.delay
        )
    else:
        parser.print_help()