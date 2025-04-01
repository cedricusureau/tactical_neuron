# src/game/systems/turn_system.py

class TurnSystem:
    def __init__(self):
        self.characters = []  # Liste des personnages dans l'ordre de jeu
        self.current_index = 0  # Index du personnage actuel

    def add_character(self, character):
        """Ajoute un personnage au système de tour"""
        if character not in self.characters:
            self.characters.append(character)

    def remove_character(self, character):
        """Retire un personnage du système de tour"""
        if character in self.characters:
            # Ajuster l'index courant si nécessaire
            current_character = self.get_current_character()

            self.characters.remove(character)

            if character == current_character:
                self.current_index = self.current_index % len(self.characters) if self.characters else 0
            elif self.characters.index(current_character) < self.current_index:
                self.current_index -= 1

    def get_current_character(self):
        """Récupère le personnage dont c'est le tour"""
        if not self.characters:
            return None
        return self.characters[self.current_index]

    def next_turn(self):
        """Passe au tour suivant"""
        if not self.characters:
            return None

        # Réinitialiser les points du personnage actuel
        current = self.get_current_character()
        current.reset_turn()

        # Passer au personnage suivant
        self.current_index = (self.current_index + 1) % len(self.characters)

        return self.get_current_character()

    def reset_all_turns(self):
        """Réinitialise tous les personnages pour une nouvelle partie"""
        for character in self.characters:
            character.reset_turn()
        self.current_index = 0