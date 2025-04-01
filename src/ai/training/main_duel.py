# src/ai/training/main_duel.py
import argparse
import os
import sys
from pathlib import Path

# Ajouter le chemin racine au PYTHONPATH
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.ai.training.train_duel import train_duel_agents, evaluate_duel


def main():
    parser = argparse.ArgumentParser(description="Entraînement et évaluation d'agents IA en duel")

    # Options générales
    parser.add_argument("--mode", choices=["train", "evaluate", "both"], default="both",
                        help="Mode d'exécution: entraînement, évaluation ou les deux")

    # Options d'entraînement
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Nombre maximum d'étapes par épisode")
    parser.add_argument("--board-size", type=int, default=10,
                        help="Taille du plateau (carré)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille du mini-batch pour l'apprentissage")
    parser.add_argument("--memory-size", type=int, default=10000,
                        help="Taille de la mémoire d'expérience")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Taux d'apprentissage")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Facteur de réduction pour les récompenses futures")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Valeur initiale d'epsilon (taux d'exploration)")
    parser.add_argument("--epsilon-end", type=float, default=0.1,
                        help="Valeur finale d'epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Taux de décroissance d'epsilon")

    # Options d'évaluation
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--warrior-model", type=str,
                        help="Chemin vers le modèle du guerrier pour l'évaluation")
    parser.add_argument("--mage-model", type=str,
                        help="Chemin vers le modèle du mage pour l'évaluation")
    parser.add_argument("--no-render", action="store_true",
                        help="Désactiver l'affichage pendant l'évaluation")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Délai entre les actions lors de l'évaluation (en secondes)")

    args = parser.parse_args()

    # Créer les dossiers nécessaires s'ils n'existent pas
    os.makedirs(os.path.join(root_dir, "data", "models"), exist_ok=True)

    # Mode entraînement
    warrior_model = None
    mage_model = None

    if args.mode in ["train", "both"]:
        print("=== Démarrage de l'entraînement ===")
        warrior_agent, mage_agent = train_duel_agents(
            episodes=args.episodes,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            memory_size=args.memory_size,
            board_width=args.board_size,
            board_height=args.board_size
        )

        # Si on fait les deux modes, utiliser les modèles qu'on vient d'entraîner
        if args.mode == "both":
            # Trouver les derniers modèles enregistrés
            model_dirs = [d for d in os.listdir(os.path.join(root_dir, "data", "models"))
                          if d.startswith("duel_training_")]
            if model_dirs:
                latest_dir = max(model_dirs)
                warrior_model = os.path.join(root_dir, "data", "models", latest_dir, "warrior_model_final.pt")
                mage_model = os.path.join(root_dir, "data", "models", latest_dir, "mage_model_final.pt")

    # Mode évaluation
    if args.mode in ["evaluate", "both"]:
        # Si on est en mode évaluation seulement, utiliser les modèles spécifiés
        if args.mode == "evaluate":
            warrior_model = args.warrior_model
            mage_model = args.mage_model

        if warrior_model and mage_model and os.path.exists(warrior_model) and os.path.exists(mage_model):
            print(f"\n=== Démarrage de l'évaluation ===")
            print(f"Modèle guerrier: {warrior_model}")
            print(f"Modèle mage: {mage_model}")

            evaluate_duel(
                warrior_model_path=warrior_model,
                mage_model_path=mage_model,
                num_episodes=args.eval_episodes,
                render=not args.no_render,
                delay=args.delay
            )
        else:
            print("Erreur: Impossible d'évaluer les agents. Modèles introuvables.")
            if args.mode == "evaluate":
                print("Veuillez spécifier les chemins vers les modèles avec --warrior-model et --mage-model")


if __name__ == "__main__":
    main()