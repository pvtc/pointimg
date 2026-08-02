# Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Ajouté

- **Anti-aliasing 4×4 supersampling** des dots (`coverage_aa` + `blend_coverage`).
  Les bords de points ne sont plus crénelés. (`src/filter/render.rs`)
- **Validation complète** des paramètres : `cols > 0`, `palette_size >= 2`,
  `variance_sensitivity ∈ [0,1]`, `max_boost >= 1.0`, bornes
  dimensions/pixels. (`src/filter/params.rs::validate_params`)
- **MSRV déclarée** : Rust 1.88 (`rust-version` dans `Cargo.toml`).
- **Découpage modulaire** de `src/filter.rs` (2072 lignes) en 10 fichiers :
  `params`, `density`, `render`, `svg`, `seedgrid`, `sampling`, `util`,
  `algorithms/{grid,kmeans,voronoi,quadtree}`, `dither`, `gamma`, `halftone`.
- **Tests d'intégration** (`tests/{pipeline,svg,reproducibility,palette,halftone}.rs`)
  et benches Criterion (`benches/filter.rs`).
- **CI GitHub Actions** réécrite : cache `Swatinem/rust-cache`, matrix
  Ubuntu/macOS/Windows, job `msrv` (1.88 + stable), job `audit`, job `docs`
  (`--cfg docsrs`), job `build` release.
- **Fond transparent** (`--bg transparent`) avec sortie RGBA via `apply_rgba`.
  Le rendu SVG omet le `<rect>` de fond en mode transparent. La prévisualisation
  GUI composite sur damier (effet Photoshop).
- **Correction gamma** (`--gamma`) : moyennes en espace linéaire
  (sRGB→linear→moyenne→sRGB) via LUT 256-entrées.
- **Niveaux de verbosité CLI** : `-v` (info), `-vv` (debug), `-q` (warnings only).
  `RUST_LOG` prioritaire via `parse_default_env`.
- **Presets TOML** : `--preset FILE` / `--save-preset FILE`.
  `FilterParams::to_toml_string` / `from_toml_str` (derives `Serialize`/`Deserialize`).
- **Dithering Floyd-Steinberg** (`--dithering`) sur palette quantifiée
  (`src/filter/dither.rs`).
- **Dossier d'exemples** `assets/examples/` avec `result.png`.
- **Badges README** : MSRV 1.88 + exemples + docs des nouveaux flags (EN/FR).

### Traitement par lot (`--input` glob/dossier, `--output` pattern)

- Expansion `--input` : fichier unique, glob (`*`/`?`/`[`), dossier (filtre images).
- Pattern `--output` avec substitutions : `{n}` (index 0-padded),
  `{stem}` (nom source sans ext.), `{name}` (nom source complet).
- Continuation du batch sur fichier défectueux (log error + suite).

### Sorties WebP/AVIF

- **WebP** inclus par défaut (extension `.webp`).
- **AVIF** opt-in : `cargo build --features avif` (pull `image/avif` / `ravif`).

### Aperçu rapide (`--preview WxH`)

- Sous-échantillonnage source (filtre Triangle, aspect ratio préservé)
  avant le pipeline. 7.7× plus rapide sur 2000×2000→300×300.

### Halftone multi-canal (`--halftone`) — Rosette CMJN et couleurs dominantes

- **`HalftoneMode`** : `Off`, `Cmyk{angles}` (15°/75°/0°/45° par défaut),
  `Dominant{n, base_angle_deg}` (N couleurs extraites via k-means sur pixels source).
- **`Screening`** : `Am` (grille rotationnée par canal, rayon = coverage)
  ou `Fm` (acceptation stochastique + jitter sub-pixel, rayon quasi constant).
- **Compositeur multiply** (`render_halftone`) : part d'un papier blanc,
  applique chaque dot en mode soustractif (Magenta + Jaune = Rouge).
- **Séparation CMYK** avec UCR partiel (retire 50% de la composante neutre
  quand C+M+Y > 0.3).
- **Angles dominants** répartis par `base + i × 180° / n` pour éviter le moiré.
- Flags CLI : `--halftone`, `--screening`, `--halftone-freq`,
  `--halftone-min-radius`, `--halftone-max-dot`.

### Corrections (`Item 22`)

- **Angle de trame `--grid-angle`** : rotation de la grille (algorithme Grid)
  autour du centre de l'image. Effet "halftone screen" expérimental.

### Refactor interne (`Items 6 + 7`)

- **`Algorithm` dérive `clap::ValueEnum`** directement — suppression du
  wrapper `AlgoArg` et des redondances conversion `From<AlgoArg>`.
- **`Args::to_filter_params()`** extraite : la construction `FilterParams`
  depuis les flags CLI est centralisée (main.rs ne fait plus que router).
  La GUI reste inchangée (elle construit `DotShape` directement depuis sliders).

### Documentation (`Item 9`)

- **`CHANGELOG.md`** au format [Keep a Changelog](https://keepachangelog.com).

### GUI (`Item 19`)

- **Undo / Redo** (`Ctrl+Z` / `Ctrl+Y` ou `Ctrl+Shift+Z`) : pile d'états
  `FilterParams` avec debounce sur drag. 50 entrées max (FIFO).
- `FilterParams: PartialEq` dérivé pour détecter changements réels.
- Indicateur visuel "Annulé." / "Refait." dans la barre de statut.

### Sauvegarde GUI multi-format

- Outlet dialog étendu : PNG / JPEG / WebP / BMP / TIFF.
- PNG/WebP/TIFF préservent l'alpha ; JPEG/BMP compositent sur `bg_color`
  si `--bg transparent`.

### Affichage GUI transparent

- Preview RGBA composite sur damier 8×8 (gris clair/gris foncé, effet Photosop)
  via `rgba_to_color_image_checker`. La zone transparente est visible entre dots.

### Routing GUI halftone/transparent

- `start_compute` route vers `apply_rgba` quand `transparent` ou `halftone != Off`
  (pas de preview progressive pour ces deux modes — single-pass).

## [0.1.0] — Version initiale publiée

- 4 algorithmes : Grid, K-means, Voronoi (Lloyd pondéré), Quadtree adaptatif.
- 4 formes de dots : Cercle, Carré, Ellipse (aspect + rotation),
  Polygone régulier (3-12 côtés).
- Density map via summed-area tables (O(W×H)).
- Spatiale accélération `SeedGrid` (hash grid 2D + early stopping).
- Export PNG + SVG vectoriel.
- GUI interactive (egui/wgpu) avec preview progressive et drag & drop.
- CLI `clap` avec tous les paramètres.
- Palette réduite optionnelle (k-means RGB sur dots).
- Fond personnalisable : white, black, ou `#rrggbb`.
- 26 tests unitaires + clippy strict.

[Unreleased]: https://github.com/pvtc/pointimg/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pvtc/pointimg/releases/tag/v0.1.0