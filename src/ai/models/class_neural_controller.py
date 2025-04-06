# src/ai/models/class_neural_controller.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Taille 128, 128
        self.output = nn.Linear(hidden_size, output_size)  # Utiliser output au lieu de fc3

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)  # Utiliser output


class ClassCharacterAI:
    def __init__(self, input_size=25, hidden_size=128, output_size=5, character_class="warrior"):
        """
        Initialise un contrôleur IA pour un personnage d'une classe spécifique
        input_size: Taille de l'état d'entrée (plateau + info personnage)
        hidden_size: Taille de la couche cachée - IMPORTANT: Doit être 128 pour correspondre aux modèles
        output_size: Nombre d'actions possibles
        character_class: La classe du personnage contrôlé (warrior, mage, archer, etc.)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.character_class = character_class
        self.hidden_size = hidden_size

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
            'character_class': self.character_class,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }, path)
        print(f"Modèle sauvegardé: {path}")
        print(f"Taille de sortie: {self.output_size}")

    def load_model(self, path):
        """Charge un modèle préentraîné avec adaptation automatique de l'architecture"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Le fichier {path} n'existe pas")

            # Charger le modèle
            checkpoint = torch.load(path)

            # Vérifier si les paramètres sont disponibles dans le checkpoint
            saved_input_size = checkpoint.get('input_size', self.input_size)
            saved_hidden_size = checkpoint.get('hidden_size', self.hidden_size)
            saved_output_size = checkpoint.get('output_size', self.output_size)

            print(
                f"Dimensions du modèle sauvegardé: input={saved_input_size}, hidden={saved_hidden_size}, output={saved_output_size}")
            print(
                f"Dimensions actuelles: input={self.input_size}, hidden={self.hidden_size}, output={self.output_size}")

            # Recréer le modèle avec les dimensions correctes si différentes
            if (saved_input_size != self.input_size or
                    saved_hidden_size != self.hidden_size or
                    saved_output_size != self.output_size):
                print("Recréation du modèle avec les dimensions sauvegardées")
                self.input_size = saved_input_size
                self.hidden_size = saved_hidden_size
                self.output_size = saved_output_size
                self.model = NeuralNetwork(saved_input_size, saved_hidden_size, saved_output_size)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            # Adapter le dictionnaire d'état pour la compatibilité avec l'ancienne architecture
            state_dict = checkpoint['model_state_dict']

            # Gérer le cas où fc3 est utilisé au lieu de output dans le modèle sauvegardé
            if 'fc3.weight' in state_dict and 'output.weight' not in state_dict:
                state_dict['output.weight'] = state_dict.pop('fc3.weight')
                state_dict['output.bias'] = state_dict.pop('fc3.bias')

            # Gérer le cas inverse
            if 'output.weight' in state_dict and 'fc3.weight' not in state_dict:
                if hasattr(self.model, 'fc3'):
                    state_dict['fc3.weight'] = state_dict.pop('output.weight')
                    state_dict['fc3.bias'] = state_dict.pop('output.bias')

            # Charger les poids en ignorant les erreurs de taille non correspondante
            self.model.load_state_dict(state_dict, strict=False)

            # Charger l'optimiseur si disponible
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Impossible de charger l'état de l'optimiseur, utilisation d'un nouvel optimiseur")
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            # Charger la classe de personnage si disponible
            if 'character_class' in checkpoint:
                self.character_class = checkpoint['character_class']

            self.model.eval()  # Passer en mode évaluation
            print(f"Modèle chargé avec succès: {path}")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            return False

    def find_latest_team_models_info():
        """Trouve les modèles d'équipe les plus récents et renvoie leurs informations"""
        models_dir = os.path.join(root_dir, "data", "models")

        # Trouver les dossiers d'entraînement
        training_dirs = glob.glob(os.path.join(models_dir, "team_training_*"))
        if not training_dirs:
            training_dirs = glob.glob(os.path.join(models_dir, "specialized_duel_*"))
            if not training_dirs:
                return {}

        latest_dir = max(training_dirs)

        # Récupérer les informations sur les modèles
        model_info = {}

        for model_type in ["warrior", "mage", "archer"]:
            model_path = os.path.join(latest_dir, f"{model_type}_model_final.pt")
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path)
                    # Extraire les informations de dimensions
                    input_size = checkpoint.get('input_size', 25)
                    hidden_size = checkpoint.get('hidden_size', 128)
                    output_size = checkpoint.get('output_size', None)

                    # Si output_size n'est pas sauvegardé, essayer de le déduire du modèle
                    if output_size is None:
                        state_dict = checkpoint['model_state_dict']
                        if 'fc3.weight' in state_dict:
                            output_size = state_dict['fc3.weight'].size(0)
                        elif 'output.weight' in state_dict:
                            output_size = state_dict['output.weight'].size(0)

                    model_info[model_type] = {
                        'path': model_path,
                        'input_size': input_size,
                        'hidden_size': hidden_size,
                        'output_size': output_size
                    }
                    print(
                        f"Informations du modèle {model_type}: in={input_size}, hidden={hidden_size}, out={output_size}")
                except Exception as e:
                    print(f"Erreur lors de l'analyse du modèle {model_type}: {str(e)}")

        return model_info