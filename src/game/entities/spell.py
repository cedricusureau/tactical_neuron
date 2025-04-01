# src/game/entities/spell.py

class Spell:
    """Classe représentant un sort ou une capacité utilisable par un personnage"""

    def __init__(self, name, description, ap_cost, min_range=1, max_range=1, damage=0,
                 healing=0, area_of_effect=0, cooldown=0):
        self.name = name
        self.description = description
        self.ap_cost = ap_cost  # Coût en points d'action
        self.min_range = min_range  # Portée minimale
        self.max_range = max_range  # Portée maximale
        self.damage = damage  # Dégâts infligés
        self.healing = healing  # Soins prodigués
        self.area_of_effect = area_of_effect  # Rayon de la zone d'effet (0 = cible unique)
        self.cooldown = cooldown  # Nombre de tours avant réutilisation
        self.current_cooldown = 0  # Cooldown actuel

    def can_cast(self, caster):
        """Vérifie si le sort peut être lancé (PA suffisants et pas en cooldown)"""
        return (caster.action_points >= self.ap_cost and
                self.current_cooldown == 0)

    def get_valid_targets(self, caster, board):
        """
        Détermine les cibles valides pour ce sort sur le plateau
        Retourne une liste de tuples (x, y)
        """
        valid_positions = []

        # Parcourir tous les points dans la portée du sort
        for y in range(board.height):
            for x in range(board.width):
                # Calculer la distance de Manhattan
                distance = abs(x - caster.x) + abs(y - caster.y)

                # Vérifier si la position est dans la portée
                if self.min_range <= distance <= self.max_range:
                    # Pour un sort de dégâts, on ne peut cibler que des cases avec des ennemis
                    if self.damage > 0:
                        character = board.get_character_at(x, y)
                        # On ne peut attaquer que les personnages qui ne sont pas dans notre équipe
                        if character and character.team != caster.team:
                            valid_positions.append((x, y))

                    # Pour un sort de soin, on ne peut cibler que des cases avec des alliés
                    elif self.healing > 0:
                        character = board.get_character_at(x, y)
                        # On ne peut soigner que les personnages de notre équipe
                        if character and character.team == caster.team:
                            valid_positions.append((x, y))

                    # Pour d'autres types de sorts (à développer plus tard)
                    else:
                        valid_positions.append((x, y))

        return valid_positions

    def get_area_of_effect(self, target_x, target_y, board):
        """
        Calcule toutes les positions affectées par le sort si lancé à la position cible
        Retourne une liste de tuples (x, y)
        """
        affected_positions = [(target_x, target_y)]  # La cible principale est toujours affectée

        # Si le sort a une zone d'effet
        if self.area_of_effect > 0:
            for y in range(board.height):
                for x in range(board.width):
                    # Calculer la distance de Manhattan par rapport à la cible principale
                    distance = abs(x - target_x) + abs(y - target_y)

                    # Ajouter si dans la zone d'effet et pas déjà dans la liste
                    if 0 < distance <= self.area_of_effect and (x, y) not in affected_positions:
                        affected_positions.append((x, y))

        return affected_positions

    def cast(self, caster, target_x, target_y, board):
        """
        Lance le sort sur la cible spécifiée
        Retourne True si le sort a été lancé avec succès
        """
        # Vérifier si le sort peut être lancé
        if not self.can_cast(caster):
            return False

        # Vérifier si la cible est valide
        valid_targets = self.get_valid_targets(caster, board)
        if (target_x, target_y) not in valid_targets:
            return False

        # Réduire les points d'action du lanceur
        caster.action_points -= self.ap_cost

        # Appliquer le cooldown
        self.current_cooldown = self.cooldown

        # Calculer la zone d'effet
        affected_positions = self.get_area_of_effect(target_x, target_y, board)

        # Appliquer les effets à toutes les positions affectées
        for pos_x, pos_y in affected_positions:
            character = board.get_character_at(pos_x, pos_y)
            if character:
                # Appliquer les dégâts aux ennemis
                if self.damage > 0 and character.team != caster.team:
                    character.take_damage(self.damage)

                # Appliquer les soins aux alliés
                if self.healing > 0 and character.team == caster.team:
                    character.heal(self.healing)

        return True

    def update_cooldown(self):
        """Met à jour le cooldown du sort à la fin d'un tour"""
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

    def __str__(self):
        """Représentation textuelle du sort pour l'affichage"""
        effect_text = ""
        if self.damage > 0:
            effect_text += f"Dégâts: {self.damage} "
        if self.healing > 0:
            effect_text += f"Soins: {self.healing} "

        cooldown_text = f"(CD: {self.current_cooldown}/{self.cooldown})" if self.cooldown > 0 else ""

        return f"{self.name} - {self.ap_cost} PA - Portée: {self.min_range}-{self.max_range} - {effect_text}{cooldown_text}"