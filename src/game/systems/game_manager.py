# src/game/systems/game_manager.py
import pygame
from ..board.grid_board import GridBoard
from ..entities.character import Character
from .turn_system import TurnSystem
from .spell_system import SpellSystem
from ...ai.models.neural_controller import CharacterAI
from ...ai.models.class_neural_controller import ClassCharacterAI
import os
from src.ai.utils.state_encoder import default_encoder

class GameManager:
    """
    Gère l'état du jeu et coordonne les différents systèmes
    """

    def __init__(self, config):
        # Configuration
        self.config = config
        self.board_width = config["board"]["width"]
        self.board_height = config["board"]["height"]
        self.cell_size = config["board"]["cell_size"]

        # Systèmes de jeu
        self.board = GridBoard(self.board_width, self.board_height)
        self.turn_system = TurnSystem()
        self.spell_system = SpellSystem(self)

        # Variables d'état
        self.selected_character = None
        self.possible_moves = []
        self.ai_delay = 0
        self.running = True
        self.game_state = "movement"  # États: movement, casting

        # Chargement des ressources
        self.init_colors()

    def init_colors(self):
        """Initialise les couleurs utilisées dans le jeu"""
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "grid": (100, 100, 100),
            "highlight": (255, 255, 0, 100),  # Jaune semi-transparent
            "possible_move": (0, 255, 0, 100),  # Vert semi-transparent
            "player": (0, 0, 255),
            "enemy": (200, 0, 0),
            "ai_controlled": (128, 0, 255),
            "cell_light": (230, 230, 230),
            "cell_dark": (200, 200, 200),
            "info_bg": (240, 240, 240)
        }

    def setup_game(self, ai_mode=False, warrior_model=None, mage_model=None, specialized=False):
        """Configure une nouvelle partie avec possibilité de modèles IA spécifiques par classe"""
        # Créer le personnage guerrier
        warrior = Character("Guerrier", "warrior", 1, 1, team=0)
        self.spell_system.assign_default_spells(warrior)

        # Créer le personnage mage
        mage = Character("Mage", "mage", self.board_width - 2, self.board_height - 2, team=1)
        self.spell_system.assign_default_spells(mage)

        # Configuration des IA
        if ai_mode:
            # Utiliser la taille d'état standardisée
            warrior_state_size = default_encoder.state_size
            mage_state_size = default_encoder.state_size

            warrior_action_size = 5 + len(warrior.spells)  # 4 directions + passer + sorts
            mage_action_size = 5 + len(mage.spells)

            # Charger le modèle du mage
            if mage_model and os.path.exists(mage_model):
                try:
                    # Utiliser la taille correcte (46 au lieu de 50)
                    mage_state_size = 46  # Taille utilisée pendant l'entraînement
                    mage_action_size = 5 + len(mage.spells)  # 4 directions + passer + sorts

                    # Créer et charger l'IA appropriée
                    if specialized:
                        mage_ai = ClassCharacterAI(mage_state_size, 128, mage_action_size, "mage")
                    else:
                        mage_ai = CharacterAI(mage_state_size, 128, mage_action_size)

                    mage_ai.load_model(mage_model)
                    mage.set_ai_controller(mage_ai)
                    print(f"Modèle mage chargé: {mage_model}")
                except Exception as e:
                    print(f"Erreur lors du chargement du modèle mage: {e}")

        # Ajouter les personnages au jeu
        self.turn_system.add_character(warrior)
        self.board.add_character(warrior)

        self.turn_system.add_character(mage)
        self.board.add_character(mage)

        print(f"Jeu initialisé: Mode {'IA' if ai_mode else 'manuel'}")

    def handle_event(self, event):
        """Gère un événement pygame"""
        # Quitter le jeu
        if event.type == pygame.QUIT:
            self.running = False
            return

        # Récupérer le personnage actif
        current_character = self.turn_system.get_current_character()
        if not current_character:
            return

        # Gestion des touches numériques pour sélectionner un sort
        if event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_9:
                spell_index = event.key - pygame.K_1  # 0-based index
                if self.spell_system.select_spell(spell_index):
                    self.game_state = "casting"
                    print(f"Sort {spell_index + 1} sélectionné")

                    # Désélectionner le personnage (désactive le mode déplacement)
                    self.selected_character = None
                    self.possible_moves = []

        # Contrôle manuel (ne fonctionne que si le personnage actif n'est pas contrôlé par l'IA)
        if not current_character.ai_controlled:
            # Gestion des clics de souris
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Clic gauche
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x = mouse_x // self.cell_size
                grid_y = mouse_y // self.cell_size

                # Vérifier que le clic est dans les limites du plateau
                if 0 <= grid_x < self.board_width and 0 <= grid_y < self.board_height:
                    # Mode de lancement de sort
                    if self.game_state == "casting":
                        # Essayer de lancer le sort sélectionné sur la cible
                        if self.spell_system.cast_selected_spell(grid_x, grid_y):
                            print(f"Sort lancé sur ({grid_x}, {grid_y})")
                            self.game_state = "movement"  # Retour au mode déplacement
                        else:
                            print("Cible invalide pour ce sort")

                    # Mode de déplacement
                    elif self.game_state == "movement":
                        # Si un personnage est sélectionné et que la case est un mouvement possible
                        if self.selected_character and (grid_x, grid_y) in self.possible_moves:
                            # Déplacer le personnage via le plateau
                            if self.board.move_character(self.selected_character, grid_x, grid_y):
                                # Recalculer les mouvements possibles avec les PM restants
                                if self.selected_character.movement_points > 0:
                                    self.possible_moves = self.board.get_possible_moves(
                                        self.selected_character,
                                        self.selected_character.movement_points
                                    )
                                else:
                                    self.selected_character = None
                                    self.possible_moves = []
                        else:
                            # Essayer de sélectionner un personnage
                            character = self.board.get_character_at(grid_x, grid_y)
                            if character and character == current_character:
                                if character.movement_points > 0:  # Ne sélectionner que si des PM sont disponibles
                                    self.selected_character = character
                                    self.possible_moves = self.board.get_possible_moves(
                                        character,
                                        character.movement_points
                                    )
                                else:
                                    print("Ce personnage n'a plus de PM disponibles!")
                            else:
                                self.selected_character = None
                                self.possible_moves = []

        # Gestion des touches clavier (toujours disponible, même en mode IA)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Fin du tour
                self.turn_system.next_turn()
                self.selected_character = None
                self.possible_moves = []
                self.spell_system.deselect_spell()
                self.game_state = "movement"

            elif event.key == pygame.K_ESCAPE:
                # Annuler la sélection actuelle (sort ou personnage)
                if self.game_state == "casting":
                    self.spell_system.deselect_spell()
                    self.game_state = "movement"
                else:
                    self.selected_character = None
                    self.possible_moves = []

            elif event.key == pygame.K_a and current_character:
                # Activer/désactiver le mode IA pour le personnage actuel
                if current_character.ai_controlled:
                    current_character.disable_ai()
                    print(f"{current_character.name} est maintenant contrôlé manuellement")
                else:
                    # Créer un contrôleur IA si nécessaire
                    if not current_character.ai_controller:
                        current_character.set_ai_controller(CharacterAI())
                    else:
                        current_character.ai_controlled = True
                    print(f"{current_character.name} est maintenant contrôlé par l'IA")

    def update(self):
        """Met à jour l'état du jeu"""
        # Contrôle IA
        current_character = self.turn_system.get_current_character()
        if current_character and current_character.ai_controlled:
            self.ai_delay += 1

            # Ajouter un délai pour que l'action de l'IA ne soit pas trop rapide
            if self.ai_delay >= 30:  # Environ 0.5 seconde à 60 FPS
                self.ai_delay = 0

                # IA simple : d'abord essayer de lancer un sort, puis bouger si possible

                # 1. Essayer de lancer un sort si des points d'action sont disponibles
                if current_character.action_points > 0:
                    # Parcourir tous les sorts disponibles
                    for spell_index, spell in enumerate(current_character.spells):
                        if spell.can_cast(current_character):
                            # Trouver des cibles valides
                            targets = spell.get_valid_targets(current_character, self.board)
                            if targets:
                                # Choisir la première cible (simple)
                                target_x, target_y = targets[0]
                                if current_character.cast_spell(spell_index, target_x, target_y, self.board):
                                    print(
                                        f"IA: {current_character.name} lance {spell.name} sur ({target_x}, {target_y})")
                                    # Vérifier si des personnages sont morts
                                    self.spell_system.check_for_casualties()
                                    break  # Sortir après avoir lancé un sort

                # 2. Essayer de se déplacer si des points de mouvement sont disponibles
                if current_character.movement_points > 0:
                    move = current_character.get_next_ai_move(self.board)

                    if move:
                        new_x, new_y = move
                        # Déplacer le personnage
                        self.board.move_character(current_character, new_x, new_y)
                        print(f"IA: {current_character.name} se déplace vers ({new_x}, {new_y})")
                    else:
                        # Si l'IA ne peut pas/ne veut pas bouger ou lancer de sort, passer au tour suivant
                        self.turn_system.next_turn()
                        print(f"IA: {current_character.name} termine son tour")
                else:
                    # Si plus de PM et plus de PA utile, passer au tour suivant
                    self.turn_system.next_turn()
                    print(f"IA: {current_character.name} termine son tour")

        # Vérifier si la partie est terminée (tous les personnages d'une équipe sont morts)
        self.check_game_end()

    def check_game_end(self):
        """Vérifie si la partie est terminée"""
        team0_alive = False
        team1_alive = False

        for character in self.board.characters:
            if character.team == 0:
                team0_alive = True
            elif character.team == 1:
                team1_alive = True

        if not team0_alive:
            print("Équipe ennemie victorieuse!")
            # Vous pourriez ajouter un écran de fin ici

        if not team1_alive:
            print("Équipe du joueur victorieuse!")
            # Vous pourriez ajouter un écran de fin ici

    def render(self, screen):
        """Dessine l'état du jeu sur l'écran"""
        # Effacer l'écran
        screen.fill(self.colors["white"])

        # Dessiner le plateau (style échiquier)
        for y in range(self.board_height):
            for x in range(self.board_width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                # Alternance des couleurs
                color = self.colors["cell_light"] if (x + y) % 2 == 0 else self.colors["cell_dark"]
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, self.colors["grid"], rect, 1)

        # Dessiner les mouvements possibles si en mode déplacement
        if self.game_state == "movement":
            for x, y in self.possible_moves:
                highlight = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                highlight.fill(self.colors["possible_move"])
                screen.blit(highlight, (x * self.cell_size, y * self.cell_size))

        # Dessiner les cibles possibles si en mode lancement de sort
        if self.game_state == "casting":
            self.spell_system.render_targets(screen)

        # Dessiner les personnages
        for character in self.board.characters:
            # Couleur différente pour le personnage sélectionné
            if character == self.selected_character:
                # Rectangle de sélection
                select_rect = pygame.Rect(
                    character.x * self.cell_size,
                    character.y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                highlight = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                highlight.fill(self.colors["highlight"])
                screen.blit(highlight, select_rect)

            # Couleur spécifique selon le type de personnage
            if character == self.turn_system.get_current_character():
                color = self.colors["ai_controlled"] if character.ai_controlled else self.colors["player"]
            else:
                color = self.colors["enemy"] if character.team == 1 else self.colors["player"]

            # Dessiner le personnage (cercle pour le moment)
            pygame.draw.circle(
                screen,
                color,
                (character.x * self.cell_size + self.cell_size // 2,
                 character.y * self.cell_size + self.cell_size // 2),
                self.cell_size // 3
            )

            # Dessiner la barre de vie
            health_percent = character.health / character.max_health
            health_width = int(self.cell_size * 0.8 * health_percent)
            health_height = 4

            # Fond de la barre de vie
            pygame.draw.rect(screen, (100, 100, 100),
                             (character.x * self.cell_size + self.cell_size * 0.1,
                              character.y * self.cell_size + self.cell_size * 0.8,
                              self.cell_size * 0.8, health_height))

            # Barre de vie active
            health_color = (0, 255, 0) if health_percent > 0.5 else (255, 255, 0) if health_percent > 0.25 else (
            255, 0, 0)
            pygame.draw.rect(screen, health_color,
                             (character.x * self.cell_size + self.cell_size * 0.1,
                              character.y * self.cell_size + self.cell_size * 0.8,
                              health_width, health_height))

            # Afficher le nom du personnage et ses PM/PA restants
            font = pygame.font.SysFont(None, 20)

            # Afficher un indicateur "IA" pour les personnages contrôlés par l'IA
            name_text = character.name
            if character.ai_controlled:
                name_text += " (IA)"

            name_surface = font.render(name_text, True, self.colors["black"])
            name_rect = name_surface.get_rect(center=(
                character.x * self.cell_size + self.cell_size // 2,
                character.y * self.cell_size + self.cell_size // 2 - 25
            ))
            screen.blit(name_surface, name_rect)

            # Afficher les PM et PA restants au-dessus du personnage
            stats_text = f"PM: {character.movement_points} | PA: {character.action_points}"
            stats_surface = font.render(stats_text, True, (0, 0, 0))
            stats_rect = stats_surface.get_rect(center=(
                character.x * self.cell_size + self.cell_size // 2,
                character.y * self.cell_size + self.cell_size // 2 - 10
            ))
            screen.blit(stats_surface, stats_rect)

        # Afficher les informations du tour
        current = self.turn_system.get_current_character()
        if current:
            # Zone d'information
            info_bg = pygame.Rect(0, self.board_height * self.cell_size,
                                  self.board_width * self.cell_size, 100)
            pygame.draw.rect(screen, self.colors["info_bg"], info_bg)
            pygame.draw.line(screen, self.colors["black"],
                             (0, self.board_height * self.cell_size),
                             (self.board_width * self.cell_size, self.board_height * self.cell_size), 2)

            # Texte d'information
            font = pygame.font.SysFont(None, 24)

            # Nom et classe du personnage actif
            turn_text = f"Tour de: {current.name} ({current.character_class})"
            if current.ai_controlled:
                turn_text += " - Contrôlé par IA"
            text_surface = font.render(turn_text, True, self.colors["black"])
            screen.blit(text_surface, (10, self.board_height * self.cell_size + 10))

            # Points de mouvement et d'action
            stats_text = f"PV: {current.health}/{current.max_health} | PA: {current.action_points}/{current.max_action_points} | PM: {current.movement_points}/{current.max_movement_points}"
            stats_surface = font.render(stats_text, True, self.colors["black"])
            screen.blit(stats_surface, (10, self.board_height * self.cell_size + 40))

            # Instructions
            help_text = "Clic: Sélectionner/Déplacer | 1-9: Sélectionner sort | Espace: Fin du tour | A: Activer/Désactiver IA"
            help_surface = font.render(help_text, True, self.colors["black"])
            screen.blit(help_surface, (10, self.board_height * self.cell_size + 70))

            # Afficher l'interface des sorts
        self.spell_system.render_spell_ui(screen)