# src/game/systems/spell_system.py
import pygame
from ..entities.spell import Spell


class SpellSystem:
    """
    Système qui gère l'utilisation des sorts dans le jeu
    """

    def __init__(self, game_manager):
        self.game_manager = game_manager
        self.selected_spell = None
        self.possible_targets = []

        # Couleurs
        self.colors = {
            "spell_bg": (50, 50, 100, 200),  # Semi-transparent
            "spell_selected": (100, 100, 200),
            "spell_available": (255, 255, 255),
            "spell_unavailable": (150, 150, 150),
            "target_select": (255, 0, 0, 100),  # Rouge semi-transparent
            "area_effect": (255, 128, 0, 70),  # Orange semi-transparent
            "spell_range": (0, 128, 255, 50),  # Bleu semi-transparent pour la portée
            "sidebar_bg": (40, 40, 80, 230)  # Fond de la barre latérale
        }

    def create_default_spells(self):
        """Crée des sorts par défaut pour les personnages"""

        # Sorts pour les guerriers
        attack = Spell("Attaque", "Une attaque de base", 2, min_range=1, max_range=1, damage=20)
        slash = Spell("Coup Puissant", "Une attaque plus puissante", 4, min_range=1, max_range=1, damage=40, cooldown=2)

        # Sorts pour les mages
        fireball = Spell("Boule de Feu", "Lance une boule de feu", 3, min_range=1, max_range=3, damage=25)
        aoe_spell = Spell("Explosion", "Explosion qui affecte une zone", 5, min_range=2, max_range=4,
                          damage=15, area_of_effect=1, cooldown=3)
        heal = Spell("Soin", "Soigne un allié", 3, min_range=0, max_range=2, healing=20)

        return {
            "warrior": [attack, slash],
            "mage": [fireball, aoe_spell, heal]
        }

    def assign_default_spells(self, character):
        """Assigne les sorts par défaut en fonction de la classe du personnage"""
        default_spells = self.create_default_spells()

        if character.character_class in default_spells:
            for spell in default_spells[character.character_class]:
                character.add_spell(spell)

    def select_spell(self, spell_index):
        """Sélectionne un sort et détermine ses cibles possibles"""
        current_character = self.game_manager.turn_system.get_current_character()
        if not current_character:
            return False

        # Désélectionner d'abord le sort actuel
        self.deselect_spell()

        # Vérifier si l'index est valide
        if 0 <= spell_index < len(current_character.spells):
            spell = current_character.spells[spell_index]

            # Vérifier si le sort peut être lancé
            if spell.can_cast(current_character):
                self.selected_spell = spell_index
                # Trouver toutes les cibles valides
                self.possible_targets = spell.get_valid_targets(current_character, self.game_manager.board)
                return True

        return False

    def deselect_spell(self):
        """Désélectionne le sort actuel"""
        self.selected_spell = None
        self.possible_targets = []

    def cast_selected_spell(self, target_x, target_y):
        """Lance le sort sélectionné sur la cible spécifiée"""
        current_character = self.game_manager.turn_system.get_current_character()
        if not current_character or self.selected_spell is None:
            return False

        # Vérifier si la cible est valide
        if (target_x, target_y) in self.possible_targets:
            # Lancer le sort
            success = current_character.cast_spell(self.selected_spell, target_x, target_y, self.game_manager.board)

            # Désélectionner le sort après l'avoir lancé
            if success:
                # Vérifier si des personnages sont morts après le lancement du sort
                self.check_for_casualties()
                self.deselect_spell()

            return success

        return False

    def check_for_casualties(self):
        """Vérifie si des personnages sont morts et les retire du jeu"""
        casualties = []
        for character in self.game_manager.board.characters:
            if character.is_dead():
                casualties.append(character)

        # Retirer les personnages morts
        for character in casualties:
            self.game_manager.board.remove_character(character)
            self.game_manager.turn_system.remove_character(character)
            print(f"{character.name} est mort !")

    def render_spell_ui(self, screen):
        """Affiche l'interface utilisateur des sorts dans une barre latérale à droite"""
        current_character = self.game_manager.turn_system.get_current_character()
        if not current_character or not current_character.spells:
            return

        # Dimensions et position de la barre latérale
        sidebar_width = 200
        cell_size = self.game_manager.cell_size
        board_width = self.game_manager.board_width * cell_size
        board_height = self.game_manager.board_height * cell_size

        # Créer une surface pour la barre latérale (pleine hauteur)
        sidebar_surface = pygame.Surface((sidebar_width, board_height), pygame.SRCALPHA)
        sidebar_surface.fill(self.colors["sidebar_bg"])
        screen.blit(sidebar_surface, (board_width, 0))

        # Titre
        font = pygame.font.SysFont(None, 24)
        title = font.render("Sorts disponibles:", True, (255, 255, 255))
        screen.blit(title, (board_width + 10, 10))

        # Liste des sorts
        spell_font = pygame.font.SysFont(None, 20)
        y_offset = 40

        for i, spell in enumerate(current_character.spells):
            # Fond du sort (différent si sélectionné)
            spell_rect = pygame.Rect(
                board_width + 10,
                y_offset + i * 60,
                sidebar_width - 20,
                55
            )

            # Couleur différente selon la disponibilité et la sélection
            if i == self.selected_spell:
                color = self.colors["spell_selected"]
            elif spell.can_cast(current_character):
                color = self.colors["spell_available"]
            else:
                color = self.colors["spell_unavailable"]

            pygame.draw.rect(screen, color, spell_rect)
            pygame.draw.rect(screen, (0, 0, 0), spell_rect, 1)  # Bordure

            # Nom du sort
            spell_name = f"{i + 1}. {spell.name}"
            name_surface = spell_font.render(spell_name, True, (0, 0, 0))
            screen.blit(name_surface, (spell_rect.x + 5, spell_rect.y + 5))

            # Coût et cooldown
            cost_text = f"Coût: {spell.ap_cost} PA"
            if spell.cooldown > 0:
                cost_text += f" - CD: {spell.current_cooldown}/{spell.cooldown}"

            cost_color = (0, 0, 0) if spell.can_cast(current_character) else (100, 100, 100)
            cost_surface = spell_font.render(cost_text, True, cost_color)
            screen.blit(cost_surface, (spell_rect.x + 5, spell_rect.y + 25))

            # Effets
            effects_text = ""
            if spell.damage > 0:
                effects_text += f"Dégâts: {spell.damage} "
            if spell.healing > 0:
                effects_text += f"Soins: {spell.healing} "

            effects_surface = spell_font.render(effects_text, True, cost_color)
            screen.blit(effects_surface, (spell_rect.x + 5, spell_rect.y + 40))

    def render_targets(self, screen):
        """Affiche les cibles possibles pour le sort sélectionné"""
        if self.selected_spell is None:
            return

        current_character = self.game_manager.turn_system.get_current_character()
        if not current_character:
            return

        spell = current_character.spells[self.selected_spell]
        cell_size = self.game_manager.cell_size

        # D'abord, afficher la portée du sort (toutes les cases dans la portée potentielle)
        self.render_spell_range(screen, current_character, spell)

        # Ensuite, afficher les cibles valides
        for x, y in self.possible_targets:
            target_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            target_surface.fill(self.colors["target_select"])
            screen.blit(target_surface, (x * cell_size, y * cell_size))

            # Si le sort a une zone d'effet, la simuler lorsque la souris survole une cible
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x, grid_y = mouse_x // cell_size, mouse_y // cell_size

            if (grid_x, grid_y) == (x, y):
                affected = spell.get_area_of_effect(x, y, self.game_manager.board)

                # Afficher la zone d'effet
                for ax, ay in affected:
                    if (ax, ay) != (x, y):  # Ne pas redessiner la cible principale
                        aoe_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                        aoe_surface.fill(self.colors["area_effect"])
                        screen.blit(aoe_surface, (ax * cell_size, ay * cell_size))

    def render_spell_range(self, screen, character, spell):
        """Affiche la portée du sort (toutes les cases dans la portée potentielle)"""
        cell_size = self.game_manager.cell_size
        board = self.game_manager.board

        # Pour chaque case du plateau
        for y in range(board.height):
            for x in range(board.width):
                # Calculer la distance de Manhattan
                distance = abs(x - character.x) + abs(y - character.y)

                # Si la case est dans la portée du sort
                if spell.min_range <= distance <= spell.max_range:
                    # Afficher une surbrillance bleue pour indiquer la portée
                    range_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                    range_surface.fill(self.colors["spell_range"])
                    screen.blit(range_surface, (x * cell_size, y * cell_size))