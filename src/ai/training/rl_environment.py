# src/ai/training/rl_environment.py
import numpy as np
import random
from ...game.board.grid_board import GridBoard
from ...game.entities.character import Character
from ...game.systems.turn_system import TurnSystem


class RLEnvironment:
    """
    Environnement d'entraînement pour l'apprentissage par renforcement
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

        # Placer le personnage contrôlé par l'IA à une position aléatoire
        x = random.randint(0, self.board_width - 1)
        y = random.randint(0, self.board_height - 1)
        self.agent = Character("Agent", "warrior", x, y)
        self.board.add_character(self.agent)
        self.turn_system.add_character(self.agent)

        # Ajouter des ennemis aléatoires
        self.enemies = []
        for i in range(2):  # On met 2 ennemis pour l'exemple
            while True:
                x = random.randint(0, self.board_width - 1)
                y = random.randint(0, self.board_height - 1)
                if self.board.is_cell_empty(x, y):
                    enemy = Character(f"Enemy {i + 1}", "mage", x, y)
                    self.board.add_character(enemy)
                    self.turn_system.add_character(enemy)
                    self.enemies.append(enemy)
                    break

        # Réinitialiser le compteur d'étapes
        self.current_step = 0

        # Renvoyer l'état initial de l'environnement
        return self._get_state()

    def step(self, action):
        """
        Exécute une action et retourne le nouvel état, la récompense et si l'épisode est terminé
        action: 0-3 pour les 4 directions, 4 pour ne rien faire
        """
        # 0: haut, 1: droite, 2: bas, 3: gauche, 4: ne rien faire
        self.current_step += 1
        done = False
        reward = 0

        # Traiter l'action du personnage IA
        if action < 4:  # Si c'est une direction
            # Calculer la nouvelle position
            dx = [0, 1, 0, -1][action]
            dy = [-1, 0, 1, 0][action]
            new_x, new_y = self.agent.x + dx, self.agent.y + dy

            # Vérifier si le mouvement est valide
            if self.board.is_valid_position(new_x, new_y) and self.board.is_cell_empty(new_x, new_y):
                # Calculer la distance pour vérifier les PM
                distance = abs(dx) + abs(dy)

                if distance <= self.agent.movement_points:
                    # Effectuer le mouvement
                    self.board.move_character(self.agent, new_x, new_y)

                    # Petite récompense pour avoir bougé
                    reward += 0.1
                else:
                    # Pénalité pour avoir essayé un mouvement impossible
                    reward -= 0.1
            else:
                # Pénalité pour avoir essayé un mouvement invalide
                reward -= 0.1

        # Si l'action est de ne rien faire ou si tous les PM sont utilisés
        if action == 4 or self.agent.movement_points == 0:
            # Fin du tour du personnage IA
            self.turn_system.next_turn()

            # Tour des ennemis (logique simplifiée pour l'exemple)
            while self.turn_system.get_current_character() != self.agent:
                current_enemy = self.turn_system.get_current_character()

                # Logique simple d'ennemi: se déplacer vers le joueur si possible
                if current_enemy.movement_points > 0:
                    # Calculer la direction vers le joueur
                    dx = 1 if self.agent.x > current_enemy.x else -1 if self.agent.x < current_enemy.x else 0
                    dy = 1 if self.agent.y > current_enemy.y else -1 if self.agent.y < current_enemy.y else 0

                    # Essayer de se déplacer vers le joueur
                    new_x, new_y = current_enemy.x + dx, current_enemy.y + dy
                    if self.board.is_valid_position(new_x, new_y) and self.board.is_cell_empty(new_x, new_y):
                        self.board.move_character(current_enemy, new_x, new_y)

                # Fin du tour de l'ennemi
                self.turn_system.next_turn()

        # Vérifier les conditions de fin
        if self.current_step >= self.max_steps:
            done = True
            reward += 1.0  # Récompense pour avoir survécu

        # Obtenir le nouvel état
        state = self._get_state()

        return state, reward, done

    def _get_state(self):
        """
        Convertit l'état actuel du jeu en un vecteur pour le réseau de neurones
        """
        # Obtenir les informations du personnage
        char_info = [
            self.agent.x / self.board_width,  # Position X normalisée
            self.agent.y / self.board_height,  # Position Y normalisée
            self.agent.movement_points / self.agent.max_movement_points,  # PM restants
            self.agent.action_points / self.agent.max_action_points,  # PA restants
            self.agent.health / self.agent.max_health  # Santé
        ]

        # Informations sur le plateau (matrice 5x5 autour du joueur)
        board_info = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = self.agent.x + dx, self.agent.y + dy

                if not self.board.is_valid_position(x, y):
                    board_info.append(-1)  # -1 pour une position hors limites
                elif not self.board.is_cell_empty(x, y):
                    # Vérifier si c'est un ami ou un ennemi
                    character = self.board.get_character_at(x, y)
                    if character in self.enemies:
                        board_info.append(2)  # 2 pour un ennemi
                    else:
                        board_info.append(1)  # 1 pour un ami (ne devrait pas arriver pour le moment)
                else:
                    board_info.append(0)  # 0 pour une case vide

        # Informations sur les ennemis
        enemy_info = []
        for enemy in self.enemies:
            if enemy in self.board.characters:  # Si l'ennemi est encore sur le plateau
                # Distance relative normalisée
                dx = (enemy.x - self.agent.x) / self.board_width
                dy = (enemy.y - self.agent.y) / self.board_height
                health = enemy.health / enemy.max_health
                enemy_info.extend([dx, dy, health])
            else:
                # Si l'ennemi n'est plus sur le plateau (mort ou retiré)
                enemy_info.extend([0, 0, 0])

        # Combiner toutes les informations
        state = char_info + board_info + enemy_info
        return np.array(state, dtype=np.float32)

    def render(self, mode='console'):
        """
        Affiche l'état actuel de l'environnement
        """
        if mode == 'console':
            # Représentation ASCII simple du plateau
            grid = [[' ' for _ in range(self.board_width)] for _ in range(self.board_height)]

            # Marquer l'agent
            grid[self.agent.y][self.agent.x] = 'A'

            # Marquer les ennemis
            for enemy in self.enemies:
                if enemy in self.board.characters:
                    grid[enemy.y][enemy.x] = 'E'

            # Afficher le plateau
            print('-' * (self.board_width + 2))
            for row in grid:
                print('|' + ''.join(row) + '|')
            print('-' * (self.board_width + 2))

            # Afficher les informations sur le tour actuel
            current = self.turn_system.get_current_character()
            print(f"Tour: {self.current_step}/{self.max_steps}")
            print(f"Joueur actif: {'Agent' if current == self.agent else 'Ennemi'}")
            print(f"Position de l'agent: ({self.agent.x}, {self.agent.y})")
            print(f"PM: {self.agent.movement_points}/{self.agent.max_movement_points}")
            print(f"PV: {self.agent.health}/{self.agent.max_health}")