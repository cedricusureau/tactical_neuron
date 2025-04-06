# src/ai/training/main_train.py
import argparse
import os
import sys
from pathlib import Path

# Ajouter le chemin racine au PYTHONPATH
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.ai.training.train_specialized import train_specialized_agents, evaluate_duel

def main():
    parser = argparse.ArgumentParser(description="Entraînement et évaluation d'agents IA spécialisés")

    # Options générales
    parser.add_argument("--mode", choices=["train", "evaluate", "both"], default="both",
                        help="Mode d'exécution: entraînement, évaluation ou les deux")

    # Options d'entraînement
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Nombre maximum d'étapes par épisode")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille du mini-batch pour l'apprentissage")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Facteur de réduction pour les récompenses futures")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Valeur initiale d'epsilon (taux d'exploration)")
    parser.add_argument("--epsilon-end", type=float, default=0.1,
                        help="Valeur finale d'epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Taux de décroissance d'epsilon")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Intervalle pour sauvegarder les modèles")

    # Options d'évaluation
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--warrior-model", type=str,
                        help="Chemin vers le modèle du guerrier pour l'évaluation")
    parser.add_argument("--mage-model", type=str,
                        help="Chemin vers le modèle du mage pour l'évaluation")
    parser.add_argument("--no-render", action="store_true",
                        help="Désactiver l'affichage pendant l'évaluation")
    parser.add_argument("--delay", type=float, default=0,
                        help="Délai entre les actions lors de l'évaluation (en secondes)")

    args = parser.parse_args()

    # Créer les dossiers nécessaires s'ils n'existent pas
    os.makedirs(os.path.join(root_dir, "data", "models"), exist_ok=True)

    # Variables pour les chemins de modèles
    warrior_model = None
    mage_model = None

    # Mode entraînement
    if args.mode in ["train", "both"]:
        print("=== Démarrage de l'entraînement ===")
        _, _, warrior_model, mage_model = train_specialized_agents(
            episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            save_interval=args.save_interval
        )

    # Mode évaluation
    if args.mode in ["evaluate", "both"]:
        # Si mode évaluation uniquement, utiliser les modèles spécifiés
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