# src/ai/utils/state_encoder.py
import torch


class GlobalStateEncoder:
    """Encodeur d'état qui fournit une connaissance complète du plateau"""

    def __init__(self, board_size=(10, 10), max_allies=4, max_enemies=4, max_spells=8):
        self.board_size = board_size
        self.max_allies = max_allies
        self.max_enemies = max_enemies
        self.max_spells = max_spells

        # Calculer la taille de l'état
        self.self_info_size = 5  # Info du personnage principal (x, y, PM, PA, PV)
        self.ally_info_size = 3 * max_allies  # (x, y, PV) pour chaque allié
        self.enemy_info_size = 3 * max_enemies  # (x, y, PV) pour chaque ennemi
        self.spell_info_size = 2 * max_spells  # (disponibilité, cooldown) pour chaque sort

        self.state_size = (
                self.self_info_size +
                self.ally_info_size +
                self.enemy_info_size +
                self.spell_info_size
        )

    def encode(self, character, board):
        """Encode l'état global du jeu"""
        state = []

        # 1. Informations sur le personnage principal
        state.extend([
            character.x / self.board_size[0],
            character.y / self.board_size[1],
            character.movement_points / character.max_movement_points,
            character.action_points / character.max_action_points,
            character.health / character.max_health
        ])

        # 2. Informations sur les alliés (hormis le personnage principal)
        allies = [c for c in board.characters if c.team == character.team and c != character]
        allies_info = []

        for ally in allies[:self.max_allies]:  # Limiter au nombre maximum d'alliés
            allies_info.extend([
                ally.x / self.board_size[0],
                ally.y / self.board_size[1],
                ally.health / ally.max_health
            ])

        # Remplir avec des zéros si moins d'alliés que le maximum
        padding = self.max_allies - len(allies)
        if padding > 0:
            allies_info.extend([0.0] * (padding * 3))

        state.extend(allies_info)

        # 3. Informations sur les ennemis
        enemies = [c for c in board.characters if c.team != character.team]
        enemies_info = []

        for enemy in enemies[:self.max_enemies]:  # Limiter au nombre maximum d'ennemis
            enemies_info.extend([
                enemy.x / self.board_size[0],
                enemy.y / self.board_size[1],
                enemy.health / enemy.max_health
            ])

        # Remplir avec des zéros si moins d'ennemis que le maximum
        padding = self.max_enemies - len(enemies)
        if padding > 0:
            enemies_info.extend([0.0] * (padding * 3))

        state.extend(enemies_info)

        # 4. Informations sur les sorts
        for i in range(self.max_spells):
            if i < len(character.spells):
                spell = character.spells[i]
                state.append(1.0 if spell.can_cast(character) else 0.0)
                state.append(spell.current_cooldown / max(1, spell.cooldown))
            else:
                state.append(0.0)
                state.append(0.0)

        return torch.FloatTensor(state)


# Configuration par défaut pour référence
DEFAULT_STATE_CONFIG = {
    "board_size": (10, 10),
    "max_allies": 4,
    "max_enemies": 4,
    "max_spells": 8,
    "action_mapping": {
        0: "move_up",
        1: "move_right",
        2: "move_down",
        3: "move_left",
        4: "wait",
        5: "spell_0",
        6: "spell_1",
        # etc.
    }
}

# Créer une instance par défaut
default_encoder = GlobalStateEncoder(**{k: v for k, v in DEFAULT_STATE_CONFIG.items()
                                        if k in ["board_size", "max_allies", "max_enemies", "max_spells"]})