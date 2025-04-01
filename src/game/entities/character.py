# src/game/entities/character.py
import torch
from src.ai.utils.state_encoder import default_encoder

class Character:
    def __init__(self, name, character_class, x, y, team=0):
        self.name = name
        self.character_class = character_class
        self.x = x
        self.y = y
        self.team = team  # 0 = joueur, 1 = ennemi

        # Statistiques de base
        self.max_health = 100
        self.health = self.max_health
        self.max_action_points = 6
        self.action_points = self.max_action_points
        self.max_movement_points = 3  # Le personnage peut se déplacer de 3 cases par tour
        self.movement_points = self.max_movement_points

        # Statistiques de combat
        self.attack = 10
        self.defense = 5

        # Liste des sorts disponibles
        self.spells = []

        # Mode de contrôle (manuel ou IA)
        self.ai_controlled = False
        self.ai_controller = None

    def set_ai_controller(self, controller):
        """
        Associe un contrôleur IA au personnage
        """
        self.ai_controller = controller
        self.ai_controlled = True

    def disable_ai(self):
        """
        Désactive le contrôle par IA
        """
        self.ai_controlled = False

    def add_spell(self, spell):
        """
        Ajoute un sort au personnage
        """
        self.spells.append(spell)

    def can_cast_spell(self, spell_index):
        """
        Vérifie si le personnage peut lancer le sort spécifié
        """
        if 0 <= spell_index < len(self.spells):
            return self.spells[spell_index].can_cast(self)
        return False

    def cast_spell(self, spell_index, target_x, target_y, board):
        """
        Lance le sort spécifié sur la cible
        """
        if 0 <= spell_index < len(self.spells):
            return self.spells[spell_index].cast(self, target_x, target_y, board)
        return False

    def move_to(self, new_x, new_y):
        """
        Déplace le personnage à une nouvelle position
        Retourne la distance parcourue ou -1 si le mouvement est impossible
        """
        # Calcul de la distance avec la distance de Manhattan
        dx = abs(new_x - self.x)
        dy = abs(new_y - self.y)
        distance = dx + dy

        if distance > self.movement_points:
            return -1  # Mouvement impossible, trop loin

        # Réduire les points de mouvement
        self.movement_points -= distance

        # La position elle-même est mise à jour par la classe GridBoard
        return distance

    def reset_turn(self):
        """Réinitialise les points d'action et de mouvement pour un nouveau tour"""
        self.action_points = self.max_action_points
        self.movement_points = self.max_movement_points

        # Mettre à jour les cooldowns des sorts
        for spell in self.spells:
            spell.update_cooldown()

    def take_damage(self, amount):
        """Inflige des dégâts au personnage"""
        # Appliquer la défense (réduction simple des dégâts)
        actual_damage = max(1, amount - self.defense // 2)
        self.health = max(0, self.health - actual_damage)
        return self.is_dead()

    def heal(self, amount):
        """Soigne le personnage"""
        old_health = self.health
        self.health = min(self.max_health, self.health + amount)
        heal_amount = self.health - old_health

    def is_dead(self):
        """Vérifie si le personnage est mort"""
        return self.health <= 0

    from src.ai.utils.state_encoder import default_encoder

    def get_next_ai_move(self, board):
        """Obtient le prochain mouvement décidé par l'IA"""
        if not self.ai_controlled or not self.ai_controller:
            return None

        # Utiliser l'encodeur global pour générer l'état
        state = default_encoder.encode(self, board)

        # Sélectionner une action
        action = self.ai_controller.select_action(state)

        # Convertir l'action en mouvement
        return self.ai_controller.action_to_move(action, self, board)