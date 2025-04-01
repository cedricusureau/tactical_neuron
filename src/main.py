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
from src.ai.models.neural_controller import CharacterAI
# Optionnel: pour les modèles spécialisés
from src.ai.models.class_neural_controller import ClassCharacterAI


def main(ai_mode=False, warrior_model=None, mage_model=None, specialized=False):
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

    # Configurer une nouvelle partie avec les modèles IA
    game_manager.setup_game(ai_mode=ai_mode, warrior_model=warrior_model, mage_model=mage_model, specialized=specialized)

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
    parser.add_argument("--warrior-model", type=str, help="Chemin vers un modèle IA pour le guerrier")
    parser.add_argument("--mage-model", type=str, help="Chemin vers un modèle IA pour le mage")
    parser.add_argument("--specialized", action="store_true", help="Utiliser les modèles spécialisés par classe")

    args = parser.parse_args()

    main(ai_mode=args.ai, warrior_model=args.warrior_model, mage_model=args.mage_model, specialized=args.specialized)