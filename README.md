# Tactical Grid Game

Un jeu de stratégie au tour par tour sur un plateau de type échiquier, inspiré de jeux comme Dofus.

## Concept du jeu

Dans Tactical Grid Game, les joueurs contrôlent des personnages sur un plateau de jeu divisé en cases. Chaque personnage dispose de points d'action (PA) et de points de mouvement (PM) à dépenser pendant son tour pour lancer des sorts et se déplacer sur le plateau.

### Caractéristiques principales

- **Gameplay au tour par tour** sur un plateau de type échiquier
- **Personnages avec classes différentes** (Guerrier, Mage, Archer, etc.)
- **Système de sorts variés** avec différentes zones d'effet, portées et effets
- **Gestion des ressources** (Points d'Action, Points de Mouvement)
- **Mode solo contre IA** et mode multijoueur local

## Architecture du projet

```
tactical_grid_game/
├── src/                     # Code source
│   ├── game/                # Logique du jeu
│   │   ├── entities/        # Personnages, sorts, etc.
│   │   ├── systems/         # Combat, déplacement, etc.
│   │   └── board/           # Plateau de jeu
│   ├── rendering/           # Rendu graphique (Pygame)
│   ├── ai/                  # Intelligence artificielle
│   │   ├── models/          # Modèles d'IA
│   │   └── strategies/      # Stratégies des adversaires
│   └── utils/               # Utilitaires
├── assets/                  # Ressources graphiques et sonores
├── configs/                 # Fichiers de configuration
├── data/                    # Données de jeu
│   ├── characters/          # Définitions des personnages
│   └── spells/              # Définitions des sorts
├── tests/                   # Tests unitaires et d'intégration
└── docs/                    # Documentation
```

## Installation

1. Cloner le dépôt
2. Installer les dépendances: `pip install -r requirements.txt`
3. Lancer le jeu: `python src/main.py`

## Technologies utilisées

- **Pygame** : Moteur de rendu et boucle de jeu
- **NumPy** : Calculs et gestion de matrices pour le plateau
- **JSON** : Stockage des configurations et données de jeu
