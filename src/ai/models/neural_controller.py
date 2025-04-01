# src/ai/models/neural_controller.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CharacterAI:
    def __init__(self, input_size=25, hidden_size=64, output_size=5):
        """
        Initialise un contrôleur IA pour un personnage
        input_size: Taille de l'état d'entrée (plateau + info personnage)
        hidden_size: Taille de la couche cachée
        output_size: Nombre d'actions possibles (haut, bas, gauche, droite, ne rien faire)
        """
        self.input_size = input_size
        self.output_size = output_size

        # Créer le réseau de neurones
        self.model = NeuralNetwork(input_size, hidden_size, output_size)

        # Pour l'exploration pendant l'apprentissage
        self.epsilon = 1.0  # Taux d'exploration initial
        self.epsilon_decay = 0.995  # Décroissance de l'exploration
        self.epsilon_min = 0.1  # Taux d'exploration minimal

        # Pour l'apprentissage
        self.memory = []  # Mémoire de replay pour le RL
        self.gamma = 0.95  # Facteur de réduction pour les récompenses futures

        # L'optimiseur (pour l'entraînement futur)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_state(self, character, board):
        """
        Convertit l'état du jeu en vecteur d'entrée pour le réseau de neurones
        """
        # 1. Information sur le personnage
        char_info = [
            character.x / board.width,  # Position X normalisée
            character.y / board.height,  # Position Y normalisée
            character.movement_points / character.max_movement_points,  # PM restants
            character.action_points / character.max_action_points,  # PA restants
            character.health / character.max_health  # Santé
        ]

        # 2. Informations sur le plateau (cellules autour du personnage)
        # Nous allons créer une matrice 5x5 centrée sur le personnage
        board_info = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = character.x + dx, character.y + dy

                # Si la position est hors limite
                if not board.is_valid_position(x, y):
                    board_info.append(-1)  # -1 pour une cellule hors limites
                # Si la cellule est occupée
                elif not board.is_cell_empty(x, y):
                    # 1 pour un allié, 2 pour un ennemi (à raffiner plus tard)
                    other_char = board.get_character_at(x, y)
                    board_info.append(1)
                # Si la cellule est vide
                else:
                    board_info.append(0)

        # Combiner les informations du personnage et du plateau
        state = char_info + board_info
        return torch.FloatTensor(state)

    def select_action(self, state, is_training=False):
        """
        Sélectionne une action basée sur l'état actuel
        is_training: Si True, utilise la politique epsilon-greedy
        """
        if is_training and random.random() < self.epsilon:
            # Exploration: action aléatoire
            return random.randint(0, self.output_size - 1)
        else:
            # Exploitation: meilleure action selon le modèle
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def action_to_move(self, action, character, board):
        """
        Convertit une action du réseau de neurones en mouvement sur le plateau
        action: indice de l'action (0: haut, 1: droite, 2: bas, 3: gauche, 4: ne rien faire)
        Retourne: (new_x, new_y) ou None si pas de mouvement
        """
        if character.movement_points <= 0:
            return None  # Pas de PM restants

        # Transformation de l'action en déplacement
        if action == 0:  # Haut
            new_x, new_y = character.x, character.y - 1
        elif action == 1:  # Droite
            new_x, new_y = character.x + 1, character.y
        elif action == 2:  # Bas
            new_x, new_y = character.x, character.y + 1
        elif action == 3:  # Gauche
            new_x, new_y = character.x - 1, character.y
        else:  # Ne rien faire
            return None

        # Vérifier si le mouvement est valide
        if board.is_valid_position(new_x, new_y) and board.is_cell_empty(new_x, new_y):
            return (new_x, new_y)

        return None

    def save_model(self, path):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Charge un modèle préentraîné"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()  # Passer en mode évaluation