# src/main.py
import pygame
import sys
import json
import os
import glob
import random
from pathlib import Path
import argparse

# Ajouter le chemin racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Imports du jeu
from src.game.systems.game_manager import GameManager
from src.game.entities.character import Character
from src.ai.models.class_neural_controller import ClassCharacterAI
from src.ai.utils.state_encoder import default_encoder


def find_latest_team_models():
    """Trouve les modèles d'équipe les plus récents"""
    models_dir = os.path.join(root_dir, "data", "models")

    # Vérifier si le dossier existe
    if not os.path.exists(models_dir):
        print("Dossier de modèles introuvable")
        return {}

    # Trouver tous les dossiers d'entraînement d'équipe
    training_dirs = glob.glob(os.path.join(models_dir, "team_training_*"))

    # Si aucun dossier d'équipe n'est trouvé, chercher les dossiers de duel
    if not training_dirs:
        training_dirs = glob.glob(os.path.join(models_dir, "specialized_duel_*"))
        if not training_dirs:
            print("Aucun dossier d'entraînement trouvé")
            return {}

    # Trouver le dossier le plus récent
    latest_dir = max(training_dirs)
    print(f"Dossier de modèles le plus récent: {latest_dir}")

    # Chercher les modèles pour chaque classe
    models = {}

    # Vérifier si c'est un dossier d'équipe ou de duel
    if "team_training" in os.path.basename(latest_dir):
        # Modèles d'équipe
        class_names = ["warrior", "mage", "archer"]
        for class_name in class_names:
            model_path = os.path.join(latest_dir, f"{class_name}_model_final.pt")
            if os.path.exists(model_path):
                print(f"Modèle {class_name} trouvé: {model_path}")
                models[class_name] = model_path
            else:
                print(f"Modèle {class_name} introuvable: {model_path}")
    else:
        # Modèles de duel
        warrior_model = os.path.join(latest_dir, "warrior_model_final.pt")
        mage_model = os.path.join(latest_dir, "mage_model_final.pt")

        if os.path.exists(warrior_model):
            print(f"Modèle guerrier trouvé: {warrior_model}")
            models["warrior"] = warrior_model

        if os.path.exists(mage_model):
            print(f"Modèle mage trouvé: {mage_model}")
            models["mage"] = mage_model

    return models


def main(team_mode=False, warrior_model=None, mage_model=None, archer_model=None, auto_find=True):
    # Si on est en mode équipe ou si aucun modèle n'est spécifié et auto_find est activé, chercher les plus récents
    models = {}

    if auto_find:
        models = find_latest_team_models()

        # Utiliser les modèles spécifiés en priorité
        if warrior_model:
            models["warrior"] = warrior_model
        if mage_model:
            models["mage"] = mage_model
        if archer_model:
            models["archer"] = archer_model

        if models:
            print(f"Utilisation des modèles auto-détectés:")
            for class_name, model_path in models.items():
                print(f"- {class_name.capitalize()}: {model_path}")

    # Initialisation de Pygame
    pygame.init()

    # Chargement de la configuration
    config_path = root_dir / "configs" / "game_config.json"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Constantes d'affichage
    BOARD_WIDTH = config["board"]["width"]
    BOARD_HEIGHT = config["board"]["height"]
    CELL_SIZE = config["board"]["cell_size"]
    SIDEBAR_WIDTH = 200  # Largeur de la barre latérale pour les sorts

    # Dimensions totales de l'écran avec la barre latérale
    SCREEN_WIDTH = BOARD_WIDTH * CELL_SIZE + SIDEBAR_WIDTH
    SCREEN_HEIGHT = BOARD_HEIGHT * CELL_SIZE + 100  # Espace supplémentaire pour l'UI
    FPS = 60

    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tactical Grid Game - Mode Équipe" if team_mode else "Tactical Grid Game")
    clock = pygame.time.Clock()

    # Créer le gestionnaire de jeu
    game_manager = GameManager(config)

    # Mode IA activé automatiquement si des modèles sont fournis
    ai_mode = bool(models)

    # Configurer une nouvelle partie avec les modèles IA
    if team_mode:
        game_manager.setup_team_game(models=models, ai_mode=ai_mode)
    else:
        game_manager.setup_game(ai_mode=ai_mode,
                                warrior_model=models.get("warrior"),
                                mage_model=models.get("mage"),
                                specialized=True)

    # Boucle principale
    while game_manager.running:
        # Gestion des événements
        for event in pygame.event.get():
            game_manager.handle_event(event)

        # Mise à jour du jeu
        game_manager.update()

        # Rendu
        game_manager.render(screen)

        # Mettre à jour l'affichage
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tactical Grid Game")
    parser.add_argument("--warrior-model", type=str, help="Chemin vers un modèle IA pour le guerrier")
    parser.add_argument("--mage-model", type=str, help="Chemin vers un modèle IA pour le mage")
    parser.add_argument("--archer-model", type=str, help="Chemin vers un modèle IA pour l'archer")
    parser.add_argument("--no-auto", action="store_true", help="Désactiver la détection automatique des modèles")
    parser.add_argument("--team-mode", action="store_true", help="Activer le mode équipe (3v3)")

    args = parser.parse_args()

    main(team_mode=args.team_mode,
         warrior_model=args.warrior_model,
         mage_model=args.mage_model,
         archer_model=args.archer_model,
         auto_find=not args.no_auto)