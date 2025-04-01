# src/ai/models/class_neural_controller.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from src.ai.utils.state_encoder import DEFAULT_STATE_CONFIG

class ClassNeuralNetwork(nn.Module):
    """
    Réseau de neurones avec des caractéristiques spécifiques à la classe du personnage
    """

    def __init__(self, input_size, hidden_size, output_size, character_class="warrior"):
        super(ClassNeuralNetwork, self).__init__()
        self.character_class = character_class

        # Couches communes
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Couches spécifiques à la classe
        if character_class == "warrior":
            # Le guerrier a une couche intermédiaire plus large pour la prise de décision tactique
            self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
            self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        else:  # mage
            # Le mage a une architecture plus profonde pour la gestion de sorts complexes
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, hidden_size // 2)

        # Couche de sortie
        self.output = nn.Linear(hidden_size if character_class == "warrior" else hidden_size // 2, output_size)

        # Fonction d'attention pour le mage (pour se concentrer sur les cibles)
        if character_class == "mage":
            self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.character_class == "warrior":
            x = F.relu(self.fc3(x))
            return self.output(x)
        else:  # mage
            # Mécanisme d'attention simplifié
            attention = torch.sigmoid(self.attention(x))
            x = x * attention

            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return self.output(x)


class ClassCharacterAI:
    """
    Contrôleur IA adapté à la classe du personnage
    """

    def __init__(self, input_size=50, hidden_size=128, output_size=10, character_class="warrior"):
        """
        Initialise un contrôleur IA pour un personnage
        character_class: "warrior" ou "mage"
        """
        self.input_size = input_size
        self.output_size = output_size
        self.character_class = character_class

        # Créer le réseau de neurones spécifique à la classe
        self.model = ClassNeuralNetwork(input_size, hidden_size, output_size, character_class)

        # Pour l'exploration pendant l'apprentissage
        self.epsilon = 1.0  # Taux d'exploration initial
        self.epsilon_decay = 0.995  # Décroissance de l'exploration
        self.epsilon_min = 0.1  # Taux d'exploration minimal

        # L'optimiseur (adaptations spécifiques à la classe)
        if character_class == "warrior":
            # Le guerrier utilise un taux d'apprentissage plus élevé pour des décisions plus directes
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:  # mage
            # Le mage utilise un taux d'apprentissage plus faible pour des stratégies plus précises
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)

    def get_state(self, character, board):
        """
        Convertit l'état du jeu en vecteur d'entrée pour le réseau de neurones,
        avec des caractéristiques spécifiques à la classe du personnage
        """
        # 1. Information sur le personnage
        char_info = [
            character.x / board.width,  # Position X normalisée
            character.y / board.height,  # Position Y normalisée
            character.movement_points / character.max_movement_points,  # PM restants
            character.action_points / character.max_action_points,  # PA restants
            character.health / character.max_health  # Santé
        ]

        # Informations spécifiques à la classe
        if self.character_class == "warrior":
            # Le guerrier s'intéresse plus à la proximité des ennemis
            char_info.append(character.attack / 20.0)  # Attaque normalisée
        else:  # mage
            # Le mage s'intéresse plus à la distance de sécurité
            char_info.append(character.defense / 10.0)  # Défense normalisée

        # 2. Informations sur le plateau (cellules autour du personnage)
        vision_range = 3 if self.character_class == "warrior" else 4  # Le mage voit plus loin

        board_info = []
        for dy in range(-vision_range, vision_range + 1):
            for dx in range(-vision_range, vision_range + 1):
                x, y = character.x + dx, character.y + dy

                # Si la position est hors limite
                if not board.is_valid_position(x, y):
                    board_info.append(-1)  # -1 pour une cellule hors limites
                # Si la cellule est occupée
                elif not board.is_cell_empty(x, y):
                    # 1 pour un allié, 2 pour un ennemi
                    other_char = board.get_character_at(x, y)
                    if other_char.team == character.team:
                        board_info.append(1)
                    else:
                        # Pour le guerrier, ajouter une information sur la santé de l'ennemi
                        if self.character_class == "warrior":
                            board_info.append(2 + other_char.health / other_char.max_health)
                        else:
                            board_info.append(2)
                # Si la cellule est vide
                else:
                    board_info.append(0)

        # 3. Informations sur les sorts disponibles
        spell_info = []
        for spell in character.spells:
            spell_info.extend([
                1.0 if spell.can_cast(character) else 0.0,  # Disponibilité
                min(1.0, spell.damage / 50.0) if spell.damage > 0 else 0.0,  # Dégâts normalisés
                min(1.0, spell.healing / 50.0) if spell.healing > 0 else 0.0,  # Soins normalisés
                min(1.0, spell.current_cooldown / max(1, spell.cooldown)),  # Cooldown normalisé
                min(1.0, spell.area_of_effect / 3.0)  # Zone d'effet normalisée
            ])

        # Combiner les informations
        state = char_info + board_info + spell_info
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

                # Pour le guerrier, favoriser les actions agressives
                if self.character_class == "warrior" and not is_training:
                    # Augmenter la valeur des actions d'attaque (supposées être les actions 5+)
                    for i in range(5, len(q_values)):
                        q_values[i] *= 1.2  # Bonus de 20% pour les attaques

                # Pour le mage, stratégie différente selon la santé
                # (supposons que l'état contient la santé à l'index 4)
                if self.character_class == "mage" and not is_training:
                    health = state[4].item()  # Santé normalisée
                    if health < 0.3:  # Santé basse
                        # Favoriser les actions de fuite ou de soin
                        q_values[0:4] *= 1.5  # Bonus pour les mouvements

                return torch.argmax(q_values).item()

    def action_to_move(self, action, character, board):
        """
        Convertit une action du réseau de neurones en mouvement sur le plateau
        action: indice de l'action
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
        else:  # Ne rien faire ou autre action
            return None

        # Vérifier si le mouvement est valide
        if board.is_valid_position(new_x, new_y) and board.is_cell_empty(new_x, new_y):
            return (new_x, new_y)

        return None

    def update_epsilon(self):
        """Décroît epsilon pour réduire l'exploration au fil du temps"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        """Sauvegarde le modèle avec la configuration d'état"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'character_class': self.character_class,
            'state_config': DEFAULT_STATE_CONFIG  # Sauvegarder la configuration
        }, path)

    def load_model(self, path):
        """Charge un modèle préentraîné"""
        checkpoint = torch.load(path)

        # Vérifier si le modèle correspond à la bonne classe
        saved_class = checkpoint.get('character_class', None)
        if saved_class and saved_class != self.character_class:
            print(
                f"Attention: Le modèle a été entraîné pour un {saved_class}, mais est chargé pour un {self.character_class}")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.model.eval()  # Passer en mode évaluation