# pointimg

[![Release](https://github.com/pvtc/pointimg/actions/workflows/release.yml/badge.svg)](https://github.com/pvtc/pointimg/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)

Filtre pointilliste qui transforme une image en une composition de points colorés de tailles variables. Chaque point prend la couleur moyenne de sa zone et son rayon est modulé par la luminance et la variance locale.

## Fonctionnalités

- **4 algorithmes** : Grille, K-means, Voronoi (Lloyd pondéré), Quadtree adaptatif
- **4 formes de points** : Cercle, Carré, Ellipse (aspect + rotation), Polygone régulier (3-12 côtés)
- **Density map** : redistribution automatique des points selon le détail local
- **Export PNG et SVG** vectoriel
- **GUI interactive** (egui/wgpu) avec preview progressive et drag & drop
- **CLI** avec tous les paramètres accessibles
- **Palette réduite** optionnelle (quantification des couleurs)
- **Fond personnalisable** : blanc, noir, ou n'importe quelle couleur `#rrggbb`

## Installation

### Binaires pré-compilés

Téléchargez les binaires depuis la page [Releases](https://github.com/pvtc/pointimg/releases) pour votre plateforme :
- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon, Universel)
- Windows (x86_64)

### Compilation depuis les sources

```bash
cargo build --release
```

Les binaires sont dans `target/release/` :
- `pointimg` -- outil CLI
- `pointimg-gui` -- application GUI (nécessite Vulkan ou OpenGL)

Pour compiler uniquement le CLI (sans les dépendances GUI) :

```bash
cargo build --release --no-default-features
```

## Utilisation CLI

```bash
# Voronoi, 1500 points, fond blanc
pointimg -i photo.jpg -o result.png --algorithm voronoi --num-points 1500

# Grille, fond noir
pointimg -i photo.jpg -o result.png --algorithm grid --cols 60 --bg black

# Fond hexadécimal personnalisé
pointimg -i photo.jpg -o result.png --bg "#1a1a2e"

# Forme carrée
pointimg -i photo.jpg -o result.png --shape square

# Hexagones
pointimg -i photo.jpg -o result.png --shape polygon --polygon-sides 6

# Export SVG
pointimg -i photo.jpg --svg --algorithm voronoi --num-points 2000
```

Voir `pointimg --help` pour la liste complète des options.

## Utilisation GUI

```bash
pointimg-gui
```

- Glisser-déposer une image ou cliquer "Ouvrir"
- Ajuster les paramètres dans le panneau gauche (recalcul automatique avec debounce)
- Exporter en PNG ou SVG

**Raccourcis clavier :**

| Raccourci | Action |
|---|---|
| `Ctrl+O` | Ouvrir une image |
| `Ctrl+S` | Sauvegarder le résultat |
| `Espace` | Relancer le calcul |

## Tests

```bash
cargo test --lib
cargo clippy -- -D warnings
```

## Débogage

Activez les logs de débogage avec la variable d'environnement `RUST_LOG` :

```bash
RUST_LOG=debug cargo run --release -- -i photo.jpg -o result.png
```

## Architecture

Voir [ARCHITECTURE.fr.md](ARCHITECTURE.fr.md) pour la documentation technique détaillée.

## Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.fr.md](CONTRIBUTING.fr.md) pour les directives.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.