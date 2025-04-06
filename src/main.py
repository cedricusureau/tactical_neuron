# src/main.py
import pygame
import sys
import json
import os
import glob
from pathlib import Path
import argparse

# Ajouter le chemin racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Imports du jeu
from src.game.systems.game_manager import GameManager
from src.ai.models.class_neural_controller import ClassCharacterAI


def find_latest_models():
    """Trouve les modèles les plus récents pour le guerrier et le mage"""
    models_dir = os.path.join(root_dir, "data", "models")

    # Vérifier si le dossier existe
    if not os.path.exists(models_dir):
        print("Dossier de modèles introuvable")
        return None, None

    # Trouver tous les dossiers d'entraînement spécialisés
    training_dirs = glob.glob(os.path.join(models_dir, "specialized_duel_*"))

    if not training_dirs:
        print("Aucun dossier d'entraînement trouvé")
        return None, None

    # Trouver le dossier le plus récent (basé sur le format de nom)
    latest_dir = max(training_dirs)
    print(f"Dossier de modèles le plus récent: {latest_dir}")

    # Chercher les modèles finaux
    warrior_model = os.path.join(latest_dir, "warrior_model_final.pt")
    mage_model = os.path.join(latest_dir, "mage_model_final.pt")

    # Vérifier que les deux modèles existent
    if not os.path.exists(warrior_model):
        print(f"Modèle de guerrier introuvable: {warrior_model}")
        warrior_model = None

    if not os.path.exists(mage_model):
        print(f"Modèle de mage introuvable: {mage_model}")
        mage_model = None

    return warrior_model, mage_model


def main(warrior_model=None, mage_model=None, auto_find=True):
    # Si aucun modèle n'est spécifié et auto_find est activé, chercher les plus récents
    if auto_find and (warrior_model is None or mage_model is None):
        auto_warrior, auto_mage = find_latest_models()

        # Utiliser les modèles spécifiés en priorité, sinon les auto-détectés
        warrior_model = warrior_model or auto_warrior
        mage_model = mage_model or auto_mage

        if warrior_model and mage_model:
            print(f"Utilisation des modèles auto-détectés:")
            print(f"- Guerrier: {warrior_model}")
            print(f"- Mage: {mage_model}")

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
    pygame.display.set_caption("Tactical Grid Game")
    clock = pygame.time.Clock()

    # Créer le gestionnaire de jeu
    game_manager = GameManager(config)

    # Mode IA activé automatiquement si des modèles sont fournis
    ai_mode = warrior_model is not None or mage_model is not None
    # Configurer une nouvelle partie avec les modèles IA
    game_manager.setup_game(ai_mode=ai_mode, warrior_model=warrior_model, mage_model=mage_model, specialized=True)

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
    parser.add_argument("--no-auto", action="store_true", help="Désactiver la détection automatique des modèles")

    args = parser.parse_args()

    main(warrior_model=args.warrior_model, mage_model=args.mage_model, auto_find=not args.no_auto)