# src/ai/training/duel_environment.py
import numpy as np
import random

import torch

from ...game.board.grid_board import GridBoard
from ...game.entities.character import Character
from ...game.systems.turn_system import TurnSystem
from ...game.systems.spell_system import SpellSystem
from ...game.entities.spell import Spell
from src.ai.utils.state_encoder import default_encoder


class DuelEnvironment:
    """
    Environnement d'entraînement pour deux agents IA s'affrontant en duel
    """

    def __init__(self, board_width=10, board_height=10, max_steps=100):
        self.board_width = board_width
        self.board_height = board_height
        self.max_steps = max_steps
        self.current_step = 0

        # Initialiser le plateau et les personnages
        self.reset()

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode d'entraînement
        """
        # Créer le plateau
        self.board = GridBoard(self.board_width, self.board_height)

        # Système de tour
        self.turn_system = TurnSystem()

        # Création du système de sorts
        self.spell_system = SpellSystem(self)

        # Placer le guerrier à une position aléatoire dans la moitié gauche du plateau
        warrior_x = random.randint(0, self.board_width // 2 - 1)
        warrior_y = random.randint(0, self.board_height - 1)
        self.warrior = Character("Warrior", "warrior", warrior_x, warrior_y, team=0)
        self.board.add_character(self.warrior)
        self.turn_system.add_character(self.warrior)
        self.spell_system.assign_default_spells(self.warrior)

        # Placer le mage à une position aléatoire dans la moitié droite du plateau
        mage_x = random.randint(self.board_width // 2, self.board_width - 1)
        mage_y = random.randint(0, self.board_height - 1)
        self.mage = Character("Mage", "mage", mage_x, mage_y, team=1)
        self.board.add_character(self.mage)
        self.turn_system.add_character(self.mage)
        self.spell_system.assign_default_spells(self.mage)

        # Réinitialiser le compteur d'étapes
        self.current_step = 0

        # État initial du jeu
        return {
            "warrior": self._get_character_state(self.warrior),
            "mage": self._get_character_state(self.mage)
        }

    def step(self, actions):
        """
        Exécute une action pour chaque agent et retourne les nouveaux états, les récompenses et si l'épisode est terminé
        actions: dictionnaire avec les actions pour le guerrier et le mage
        """
        self.current_step += 1
        done = False

        # Récompenses de base (rien ou presque)
        rewards = {"warrior": 0.0, "mage": 0.0}

        # Récupérer le personnage dont c'est le tour
        current_character = self.turn_system.get_current_character()
        current_type = "warrior" if current_character == self.warrior else "mage"
        enemy = self.mage if current_type == "warrior" else self.warrior

        # Sauvegarder l'état de santé de l'ennemi avant l'action
        enemy_health_before = enemy.health

        # Récupérer l'action du personnage actif
        action = actions.get(current_type)

        # Forcer un déplacement si possible
        moved = False
        if current_character.movement_points > 0:
            # Essayer l'action de déplacement choisie ou des alternatives si celle-ci ne fonctionne pas
            directions_to_try = [action] if action < 4 else [0, 1, 2,
                                                             3]  # Si action n'est pas un déplacement, essayer toutes les directions

            for direction in directions_to_try:
                dx = [0, 1, 0, -1][direction]
                dy = [-1, 0, 1, 0][direction]
                new_x, new_y = current_character.x + dx, current_character.y + dy

                if self.board.is_valid_position(new_x, new_y) and self.board.is_cell_empty(new_x, new_y):
                    if self.board.move_character(current_character, new_x, new_y):
                        moved = True
                        break  # Sortir de la boucle dès qu'un mouvement valide est effectué

        # Si le personnage est déjà bloqué et ne peut pas bouger, ne pas le pénaliser

        # Action originale (lancer un sort ou ne rien faire)
        if action >= 5 and action < 5 + len(current_character.spells):  # Lancer un sort
            spell_index = action - 5
            spell = current_character.spells[spell_index]

            if spell.can_cast(current_character):
                # Trouver les cibles valides
                targets = spell.get_valid_targets(current_character, self.board)

                if targets:
                    # Pour l'IA, cibler le personnage ennemi le plus proche pour attaque ou soi-même pour soin
                    if spell.damage > 0:
                        # Trouver la cible la plus proche de l'ennemi pour l'attaque
                        best_target = min(targets, key=lambda pos: abs(pos[0] - enemy.x) + abs(pos[1] - enemy.y))
                    else:
                        # Pour les sorts de soin, cibler soi-même
                        best_target = (current_character.x, current_character.y)

                    # Lancer le sort
                    current_character.cast_spell(spell_index, best_target[0], best_target[1], self.board)

        # Passer au tour suivant
        self.turn_system.next_turn()

        # Vérifier si un personnage est mort
        if self.warrior.is_dead():
            # Le mage gagne
            rewards["mage"] = 1.0  # Grande récompense pour la victoire
            rewards["warrior"] = -1.0  # Pénalité pour la défaite
            done = True
        elif self.mage.is_dead():
            # Le guerrier gagne
            rewards["warrior"] = 1.0  # Grande récompense pour la victoire
            rewards["mage"] = -1.0  # Pénalité pour la défaite
            done = True

        # Si le nombre max d'étapes est atteint, c'est un match nul
        if self.current_step >= self.max_steps and not done:
            done = True
            # Légère pénalité pour les deux (encourager à finir le combat)
            rewards["warrior"] = -0.5
            rewards["mage"] = -0.5

        # Obtenir les nouveaux états
        states = {
            "warrior": self._get_character_state(self.warrior),
            "mage": self._get_character_state(self.mage)
        }

        return states, rewards, done

    def _get_character_state(self, character):
        """Crée un vecteur d'état pour un personnage spécifique en utilisant l'encodeur global"""
        if character.is_dead():
            # Si le personnage est mort, renvoyer un état "nul" de la bonne taille
            return torch.zeros(default_encoder.state_size, dtype=torch.float32)

        # Utiliser l'encodeur global
        return default_encoder.encode(character, self.board)

    def get_action_space_size(self, character_type):
        """
        Retourne la taille de l'espace d'action pour un type de personnage
        """
        character = self.warrior if character_type == "warrior" else self.mage
        # 4 directions + ne rien faire + nombre de sorts
        return 5 + len(character.spells)

    def render(self, mode='console'):
        """
        Affiche l'état actuel de l'environnement
        """
        if mode == 'console':
            # Représentation ASCII simple du plateau
            grid = [[' ' for _ in range(self.board_width)] for _ in range(self.board_height)]

            # Marquer le guerrier et le mage
            if self.warrior in self.board.characters:
                grid[self.warrior.y][self.warrior.x] = 'W'
            if self.mage in self.board.characters:
                grid[self.mage.y][self.mage.x] = 'M'

            # Afficher le plateau
            print('-' * (self.board_width + 2))
            for row in grid:
                print('|' + ''.join(row) + '|')
            print('-' * (self.board_width + 2))

            # Afficher des informations supplémentaires
            print(f"Tour: {self.current_step}/{self.max_steps}")
            current = self.turn_system.get_current_character()
            current_type = "Guerrier" if current == self.warrior else "Mage"
            print(f"Tour de: {current_type}")

            if self.warrior in self.board.characters:
                print(
                    f"Guerrier: PV={self.warrior.health}/{self.warrior.max_health} PA={self.warrior.action_points} PM={self.warrior.movement_points} Pos=({self.warrior.x}, {self.warrior.y})")
            if self.mage in self.board.characters:
                print(
                    f"Mage: PV={self.mage.health}/{self.mage.max_health} PA={self.mage.action_points} PM={self.mage.movement_points} Pos=({self.mage.x}, {self.mage.y})")