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

        rewards = {"warrior": 0, "mage": 0}

        # Récupérer le personnage dont c'est le tour
        current_character = self.turn_system.get_current_character()
        current_type = "warrior" if current_character == self.warrior else "mage"

        # Récupérer l'action du personnage actif
        action = actions.get(current_type)

        if action is not None:
            # Traiter l'action selon son type
            if action < 4:  # Mouvement: haut, droite, bas, gauche
                # Calculer la nouvelle position
                dx = [0, 1, 0, -1][action]
                dy = [-1, 0, 1, 0][action]
                new_x, new_y = current_character.x + dx, current_character.y + dy

                # Vérifier si le mouvement est valide
                if self.board.is_valid_position(new_x, new_y) and self.board.is_cell_empty(new_x, new_y):
                    if self.board.move_character(current_character, new_x, new_y):
                        # Petite récompense pour avoir bougé
                        rewards[current_type] += 0.1
                    else:
                        # Pénalité pour avoir essayé un mouvement impossible
                        rewards[current_type] -= 0.1
                else:
                    # Pénalité pour avoir essayé un mouvement invalide
                    rewards[current_type] -= 0.1

            elif action == 4:  # Ne rien faire (passer)
                pass

            elif action >= 5 and action < 5 + len(current_character.spells):  # Lancer un sort
                spell_index = action - 5
                spell = current_character.spells[spell_index]

                if spell.can_cast(current_character):
                    # Trouver les cibles valides
                    targets = spell.get_valid_targets(current_character, self.board)

                    if targets:
                        # Pour l'IA, cibler le personnage ennemi le plus proche
                        enemy = self.mage if current_character == self.warrior else self.warrior

                        # Trouver la cible la plus proche de l'ennemi
                        best_target = min(targets, key=lambda pos: abs(pos[0] - enemy.x) + abs(pos[1] - enemy.y))

                        # Lancer le sort
                        if current_character.cast_spell(spell_index, best_target[0], best_target[1], self.board):
                            # Récompense pour avoir lancé un sort
                            rewards[current_type] += 0.5

                            # Vérifier si le sort a blessé l'ennemi
                            enemy_health_before = enemy.health
                            spell.cast(current_character, best_target[0], best_target[1], self.board)
                            damage_dealt = enemy_health_before - enemy.health

                            if damage_dealt > 0:
                                rewards[current_type] += damage_dealt / enemy.max_health

                            # Vérifier si l'ennemi est mort
                            if enemy.is_dead():
                                rewards[current_type] += 10.0  # Forte récompense pour une victoire
                                done = True
                    else:
                        # Pénalité pour avoir essayé de lancer un sort sans cible valide
                        rewards[current_type] -= 0.2
                else:
                    # Pénalité pour avoir essayé de lancer un sort indisponible
                    rewards[current_type] -= 0.2

        # Passer au tour suivant
        self.turn_system.next_turn()

        # Vérifier si le nombre maximal d'étapes est atteint
        if self.current_step >= self.max_steps:
            done = True

            # Petite récompense pour les deux s'ils ont survécu
            rewards["warrior"] += 1.0
            rewards["mage"] += 1.0

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