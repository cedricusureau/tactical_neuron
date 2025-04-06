# src/ai/training/team_environment.py
import numpy as np
import random
import torch

from ...game.board.grid_board import GridBoard
from ...game.entities.character import Character
from ...game.systems.turn_system import TurnSystem
from ...game.systems.spell_system import SpellSystem
from src.ai.utils.state_encoder import default_encoder


class TeamEnvironment:
    """
    Environnement d'entraînement pour des combats en équipe (3v3) avec apprentissage autonome
    """

    def __init__(self, board_width=12, board_height=12, max_steps=50):
        self.board_width = board_width
        self.board_height = board_height
        self.max_steps = max_steps
        self.current_step = 0

        # Classes possibles
        self.character_classes = ["warrior", "mage", "archer"]

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

        # Créer les équipes
        self.team0 = []  # Équipe du joueur
        self.team1 = []  # Équipe ennemie

        # Assigner aléatoirement les classes pour chaque équipe
        team0_classes = random.choices(self.character_classes, k=3)
        team1_classes = random.choices(self.character_classes, k=3)

        # Créer les personnages de l'équipe 0 (dans la partie gauche du plateau)
        for i, char_class in enumerate(team0_classes):
            # Positionner aléatoirement dans la zone gauche
            x = random.randint(1, self.board_width // 3 - 1)
            y = random.randint(1 + i * 3, 3 + i * 3)  # Répartir verticalement

            character = Character(f"Team0_{char_class}_{i}", char_class, x, y, team=0)
            self.board.add_character(character)
            self.turn_system.add_character(character)
            self.spell_system.assign_default_spells(character)
            self.team0.append(character)

        # Créer les personnages de l'équipe 1 (dans la partie droite du plateau)
        for i, char_class in enumerate(team1_classes):
            # Positionner aléatoirement dans la zone droite
            x = random.randint(self.board_width * 2 // 3, self.board_width - 2)
            y = random.randint(1 + i * 3, 3 + i * 3)  # Répartir verticalement

            character = Character(f"Team1_{char_class}_{i}", char_class, x, y, team=1)
            self.board.add_character(character)
            self.turn_system.add_character(character)
            self.spell_system.assign_default_spells(character)
            self.team1.append(character)

        # Réinitialiser le compteur d'étapes
        self.current_step = 0

        # Attribuer les ID aux personnages pour l'indexation
        for i, character in enumerate(self.team0 + self.team1):
            character.id = i

        # Composition des équipes pour l'observation
        self.team0_comp = team0_classes
        self.team1_comp = team1_classes

        # État initial du jeu - dictionnaire avec un état pour chaque agent
        return self._get_all_states()

    def _get_all_states(self):
        """Obtenir les états pour tous les agents actifs"""
        states = {}

        # États pour l'équipe 0
        for character in self.team0:
            if character in self.board.characters:  # Si toujours vivant
                states[f"team0_{character.id}"] = self._get_character_state(character)

        # États pour l'équipe 1
        for character in self.team1:
            if character in self.board.characters:  # Si toujours vivant
                states[f"team1_{character.id}"] = self._get_character_state(character)

        return states

    def step(self, actions):
        """
        Exécute une action pour l'agent actif et retourne les nouveaux états, récompenses et fin d'épisode
        actions: dictionnaire avec l'action pour l'agent actif
        """
        self.current_step += 1
        done = False

        # Récompenses initiales (rien)
        rewards = {}
        for character in self.team0 + self.team1:
            if character in self.board.characters:  # Si toujours vivant
                rewards[f"team0_{character.id}" if character in self.team0 else f"team1_{character.id}"] = 0.0

        # Récupérer le personnage dont c'est le tour
        current_character = self.turn_system.get_current_character()
        if not current_character:
            # Si pas de personnage actif, passer au tour suivant
            self.turn_system.next_turn()
            current_character = self.turn_system.get_current_character()

        if not current_character:
            # Si toujours pas de personnage actif, c'est que tous sont morts
            done = True
            return self._get_all_states(), rewards, done

        # Déterminer l'ID de l'agent actif
        current_id = f"team0_{current_character.id}" if current_character in self.team0 else f"team1_{current_character.id}"

        # Déterminer l'équipe opposée pour les récompenses
        current_team = self.team0 if current_character in self.team0 else self.team1
        enemy_team = self.team1 if current_character in self.team0 else self.team0

        # Sauvegarder l'état de santé des personnages avant l'action pour calculer les dommages/soins
        health_before = {}
        for character in self.board.characters:
            health_before[character] = character.health

        # Récupérer l'action choisie pour le personnage actif
        action = actions.get(current_id)

        if action is not None:
            # Calculer la distance à l'ennemi le plus proche avant l'action
            closest_enemy_distance_before = float('inf')
            for enemy in enemy_team:
                if enemy in self.board.characters:
                    dist = abs(current_character.x - enemy.x) + abs(current_character.y - enemy.y)
                    closest_enemy_distance_before = min(closest_enemy_distance_before, dist)

            # Exécuter l'action
            if action < 4:  # Mouvement: haut, droite, bas, gauche
                # Calculer la nouvelle position
                dx = [0, 1, 0, -1][action]
                dy = [-1, 0, 1, 0][action]
                new_x, new_y = current_character.x + dx, current_character.y + dy

                # Tenter le déplacement
                if self.board.is_valid_position(new_x, new_y) and self.board.is_cell_empty(new_x, new_y):
                    if self.board.move_character(current_character, new_x, new_y):
                        # Récompense pour avoir bougé (base)
                        rewards[current_id] += 0.01

                        # Calculer la nouvelle distance à l'ennemi le plus proche
                        closest_enemy_distance_after = float('inf')
                        for enemy in enemy_team:
                            if enemy in self.board.characters:
                                dist = abs(current_character.x - enemy.x) + abs(current_character.y - enemy.y)
                                closest_enemy_distance_after = min(closest_enemy_distance_after, dist)

                        # Récompense différente selon la classe et la distance
                        if current_character.character_class == "warrior":
                            # Guerrier: récompensé pour se rapprocher
                            if closest_enemy_distance_after < closest_enemy_distance_before:
                                rewards[current_id] += 0.05
                            # Bonus pour être au corps à corps (distance 1)
                            if closest_enemy_distance_after == 1:
                                rewards[current_id] += 0.1

                        elif current_character.character_class in ["mage", "archer"]:
                            # Mage/archer: récompensé pour maintenir une distance idéale (entre 2 et 4)
                            ideal_distance = 3
                            if 2 <= closest_enemy_distance_after <= 4:
                                rewards[current_id] += 0.05
                            # Pénalité pour être trop proche
                            if closest_enemy_distance_after == 1:
                                rewards[current_id] -= 0.05

            elif action == 4:  # Ne rien faire
                # Légère pénalité pour ne rien faire si d'autres actions sont possibles
                if current_character.movement_points > 0 or current_character.action_points > 0:
                    rewards[current_id] -= 0.05

            elif action >= 5 and action < 5 + len(current_character.spells):  # Lancer un sort
                spell_index = action - 5
                spell = current_character.spells[spell_index]

                if spell.can_cast(current_character):
                    # Récompense de base pour avoir utilisé un sort qui peut être lancé
                    rewards[current_id] += 0.03

                    # Obtenir les cibles valides pour ce sort
                    targets = spell.get_valid_targets(current_character, self.board)

                    if targets:
                        # Sélectionner une cible de manière intelligente
                        if spell.damage > 0:  # Sort d'attaque
                            # Trouver les ennemis à portée
                            enemy_positions = []
                            for enemy in enemy_team:
                                if enemy in self.board.characters:
                                    for target_pos in targets:
                                        if abs(target_pos[0] - enemy.x) + abs(
                                                target_pos[1] - enemy.y) <= spell.area_of_effect:
                                            enemy_positions.append(target_pos)

                            if enemy_positions:
                                # Choisir une position qui affecte un ennemi
                                target_x, target_y = random.choice(enemy_positions)
                            else:
                                # Sinon, choisir une cible aléatoire
                                target_x, target_y = random.choice(targets)

                        elif spell.healing > 0:  # Sort de soin
                            # Déterminer s'il y a des alliés blessés
                            injured_allies = [ally for ally in current_team if ally in self.board.characters
                                              and ally.health < ally.max_health * 0.8]

                            if injured_allies and current_character.health >= current_character.max_health * 0.5:
                                # Si des alliés sont blessés et qu'on est en assez bonne santé, essayer de les cibler
                                ally_positions = []
                                for ally in injured_allies:
                                    for target_pos in targets:
                                        if abs(target_pos[0] - ally.x) + abs(
                                                target_pos[1] - ally.y) <= spell.area_of_effect:
                                            ally_positions.append(target_pos)

                                if ally_positions:
                                    target_x, target_y = random.choice(ally_positions)
                                else:
                                    # Sinon se soigner soi-même
                                    target_x, target_y = current_character.x, current_character.y
                            else:
                                # Se soigner soi-même si on est blessé
                                if current_character.health < current_character.max_health * 0.8:
                                    target_x, target_y = current_character.x, current_character.y
                                else:
                                    # Sinon choisir une cible aléatoire
                                    target_x, target_y = random.choice(targets)
                        else:
                            # Pour d'autres types de sorts
                            target_x, target_y = random.choice(targets)

                        # Lancer le sort
                        current_character.cast_spell(spell_index, target_x, target_y, self.board)

        # Calculer les récompenses basées sur les dommages/soins après l'action
        for character in self.board.characters:
            char_id = f"team0_{character.id}" if character in self.team0 else f"team1_{character.id}"
            health_diff = character.health - health_before.get(character, 0)

            if health_diff < 0:  # Dommages subis
                # Pénalité pour avoir subi des dommages (pour l'équipe du personnage)
                rewards[char_id] -= 0.1 * (abs(health_diff) / character.max_health)

                # Récompense pour avoir infligé des dommages (pour l'équipe adverse)
                for enemy_id in rewards.keys():
                    if (enemy_id.startswith("team0") and character in self.team1) or \
                            (enemy_id.startswith("team1") and character in self.team0):
                        rewards[enemy_id] += 0.2 * (abs(health_diff) / character.max_health)

            elif health_diff > 0:  # Soins reçus
                # Récompense pour avoir reçu des soins (seulement si nécessaire)
                if health_before.get(character, 0) < character.max_health * 0.8:
                    rewards[char_id] += 0.1 * (health_diff / character.max_health)

                    # Récompense pour avoir soigné (pour l'équipe du personnage)
                    for ally_id in rewards.keys():
                        if (ally_id.startswith("team0") and character in self.team0) or \
                                (ally_id.startswith("team1") and character in self.team1):
                            if ally_id != char_id:  # Ne pas compter deux fois
                                rewards[ally_id] += 0.05 * (health_diff / character.max_health)

        # Vérifier les décès après les actions
        for character in list(self.board.characters):
            if character.is_dead():
                # Fortes récompenses pour avoir éliminé un ennemi
                for ally_id in rewards.keys():
                    if (ally_id.startswith("team0") and character in self.team1) or \
                            (ally_id.startswith("team1") and character in self.team0):
                        rewards[ally_id] += 0.5

                # Retirer le personnage mort
                self.board.remove_character(character)
                self.turn_system.remove_character(character)

        # Passer au tour suivant
        self.turn_system.next_turn()

        # Vérifier si une équipe est éliminée
        team0_alive = any(character in self.board.characters for character in self.team0)
        team1_alive = any(character in self.board.characters for character in self.team1)

        if not team0_alive or not team1_alive:
            done = True

            # Attribuer les récompenses selon le vainqueur
            if not team0_alive and team1_alive:
                # L'équipe 1 a gagné
                for character in self.team1:
                    rewards[f"team1_{character.id}"] = 1.0
                for character in self.team0:
                    rewards[f"team0_{character.id}"] = -1.0

            elif not team1_alive and team0_alive:
                # L'équipe 0 a gagné
                for character in self.team0:
                    rewards[f"team0_{character.id}"] = 1.0
                for character in self.team1:
                    rewards[f"team1_{character.id}"] = -1.0

        # Si le nombre maximum d'étapes est atteint, c'est un match nul
        if self.current_step >= self.max_steps and not done:
            done = True

            # Légère pénalité pour tous (encourager à finir le combat)
            for character in self.team0 + self.team1:
                rewards[f"team0_{character.id}" if character in self.team0 else f"team1_{character.id}"] = -0.5

        # Obtenir les nouveaux états
        states = self._get_all_states()

        return states, rewards, done

    def _get_character_state(self, character):
        """Crée un vecteur d'état pour un personnage spécifique"""
        if character.is_dead():
            # Si le personnage est mort, renvoyer un état "nul" de la bonne taille
            return torch.zeros(default_encoder.state_size, dtype=torch.float32)

        # Utiliser l'encodeur global
        return default_encoder.encode(character, self.board)

    def get_action_space_size(self, character_class):
        """
        Retourne la taille de l'espace d'action pour une classe de personnage
        """
        # Créer un personnage temporaire pour obtenir le nombre de sorts
        temp_character = Character("temp", character_class, 0, 0)
        self.spell_system.assign_default_spells(temp_character)

        # 4 directions + ne rien faire + nombre de sorts
        return 5 + len(temp_character.spells)

    def render(self, mode='console'):
        """
        Affiche l'état actuel de l'environnement
        """
        if mode == 'console':
            # Représentation ASCII simple du plateau
            grid = [[' ' for _ in range(self.board_width)] for _ in range(self.board_height)]

            # Marquer les personnages sur le plateau
            for char in self.board.characters:
                if char in self.team0:
                    grid[char.y][char.x] = 'A'  # Équipe A (0)
                else:
                    grid[char.y][char.x] = 'B'  # Équipe B (1)

            # Afficher le plateau
            print('-' * (self.board_width + 2))
            for row in grid:
                print('|' + ''.join(row) + '|')
            print('-' * (self.board_width + 2))

            # Afficher des informations supplémentaires
            print(f"Tour: {self.current_step}/{self.max_steps}")

            current = self.turn_system.get_current_character()
            if current:
                current_team = "A" if current in self.team0 else "B"
                print(f"Tour de: Équipe {current_team} - {current.name} ({current.character_class})")

            # Afficher les personnages vivants par équipe
            print("\nÉquipe A:")
            for char in self.team0:
                if char in self.board.characters:
                    print(
                        f"  {char.name}: PV={char.health}/{char.max_health} PA={char.action_points} PM={char.movement_points} Pos=({char.x}, {char.y})")
                else:
                    print(f"  {char.name}: MORT")

            print("\nÉquipe B:")
            for char in self.team1:
                if char in self.board.characters:
                    print(
                        f"  {char.name}: PV={char.health}/{char.max_health} PA={char.action_points} PM={char.movement_points} Pos=({char.x}, {char.y})")
                else:
                    print(f"  {char.name}: MORT")