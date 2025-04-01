# src/game/board/grid_board.py
import numpy as np


class GridBoard:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Tableau pour stocker les entités sur le plateau
        self.grid = np.zeros((height, width), dtype=object)
        self.characters = []  # Liste pour garder une référence à tous les personnages

    def is_valid_position(self, x, y):
        """Vérifie si une position est à l'intérieur des limites du plateau"""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_cell_empty(self, x, y):
        """Vérifie si une cellule est vide (ne contient pas de personnage)"""
        if not self.is_valid_position(x, y):
            return False
        return self.grid[y, x] is None or self.grid[y, x] == 0

    def add_character(self, character):
        """Ajoute un personnage au plateau"""
        if self.is_valid_position(character.x, character.y) and self.is_cell_empty(character.x, character.y):
            self.grid[character.y, character.x] = character
            if character not in self.characters:
                self.characters.append(character)
            return True
        return False

    def remove_character(self, character):
        """Retire un personnage du plateau"""
        if character in self.characters:
            self.grid[character.y, character.x] = None
            self.characters.remove(character)
            return True
        return False

    def move_character(self, character, new_x, new_y):
        """Déplace un personnage vers une nouvelle position"""

        if not self.is_valid_position(new_x, new_y) or not self.is_cell_empty(new_x, new_y):
            return False

        # Calculer la distance pour vérifier les PM
        dx = abs(new_x - character.x)
        dy = abs(new_y - character.y)
        distance = dx + dy

        if distance > character.movement_points:
            return False

        # Effacer l'ancienne position
        self.grid[character.y, character.x] = None

        # Mettre à jour la position du personnage
        old_x, old_y = character.x, character.y
        character.x = new_x
        character.y = new_y

        # Définir la nouvelle position
        self.grid[new_y, new_x] = character

        # Déduire les points de mouvement
        character.movement_points -= distance

        return True

    def get_character_at(self, x, y):
        """Récupère le personnage à une position donnée"""
        if not self.is_valid_position(x, y):
            return None
        return self.grid[y, x] if self.grid[y, x] != 0 else None

    def get_possible_moves(self, character, movement_range=None):
        """Récupère toutes les positions où un personnage peut se déplacer"""
        if movement_range is None:
            movement_range = character.movement_points

        print(f"Calcul des mouvements possibles pour {character.name} avec {movement_range} PM")

        possible_moves = []

        # Recherche en largeur pour trouver toutes les positions accessibles
        visited = set()
        queue = [(character.x, character.y, 0)]  # x, y, distance

        while queue:
            x, y, dist = queue.pop(0)

            if (x, y) in visited:
                continue

            visited.add((x, y))

            # Si c'est une position valide (non occupée) et pas la position initiale
            if dist > 0 and self.is_cell_empty(x, y):
                possible_moves.append((x, y))

            # Si on n'a pas atteint la limite de mouvement, explorer les voisins
            if dist < movement_range:
                # Directions: droite, gauche, haut, bas
                directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if self.is_valid_position(nx, ny) and (nx, ny) not in visited:
                        queue.append((nx, ny, dist + 1))

        return possible_moves