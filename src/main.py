# src/main.py
import pygame
import sys
import json
import os
from pathlib import Path
import argparse

# Ajouter le chemin racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Imports du jeu
from src.game.systems.game_manager import GameManager


def main(ai_mode=False, model_path=None):
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

    # Configurer une nouvelle partie
    game_manager.setup_game(ai_mode=ai_mode, model_path=model_path)

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
    parser.add_argument("--ai", action="store_true", help="Activer le mode IA")
    parser.add_argument("--model", type=str, help="Chemin vers un modèle IA préentraîné")

    args = parser.parse_args()

    main(ai_mode=args.ai, model_path=args.model)