# Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Ajouté

- **Profils colorimétriques ICC** : détection automatique des profils embarqués,
  conversion d'entrée vers sRGB avec `moxcms`, support de `--input-profile`
  (`auto`, `srgb`, `display-p3` ou fichier `.icc`) et de `--output-profile`
  avec embedding ICC dans les sorties PNG/JPEG/WebP/TIFF.
- **Downscale préventif** : les images dépassant 8 millions de pixels ou 65 535
  pixels par côté sont réduites automatiquement en conservant leur ratio, avant
  les transformations colorimétriques coûteuses. La GUI affiche l'avertissement
  et une estimation de la mémoire de travail ; le CLI journalise ces informations.
- **Halftone CMYK haute précision** : le chemin `--gamma` utilise des couvertures
  `f32` après linéarisation pour le halftone CMJN, sans imposer un pipeline `f32`
  plus coûteux aux algorithmes pointillistes classiques.
- **Historique GUI explicite** : Undo/Redo affiche désormais les actions
  enregistrées (algorithme, nombre de points, palette, fond, gamma, etc.).
- **Contrôle couplé des rayons** : les curseurs Min/Max de la GUI sont regroupés
  et se bornent mutuellement afin d'empêcher `min_radius > max_radius`.
- **Smoke fuzzing CI** : les cibles image et preset sont compilées et exécutées
  avec 100 runs à chaque vérification CI ; elles ont également été exécutées
  localement avec nightly.
- **Audit planifié** : un workflow hebdomadaire `cargo audit` a été ajouté.
- **Presets versionnés** : les presets générés utilisent `preset_version = 1`
  et une table `[params]`. Les anciens presets TOML plats restent lisibles et
  les versions futures inconnues sont refusées explicitement.
- **Protection des workers GUI** : les calculs et density maps sont associés à
  une génération. Un worker annulé ou obsolète ne peut plus publier un résultat,
  une erreur ou une density map à la place d'un calcul plus récent.
- **Benchmarks de performance** : mesures Criterion ajoutées pour le halftone
  CMYK/AM et le chemin RGBA avec palette+dithering, avec débit exprimé en pixels.
- **Fuzzing image renforcé** : la cible fuzz utilise l'API `image::Limits`
  compatible avec les versions actuelles de la crate.
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

### Corrigé

- **Résultats GUI obsolètes** : un worker lié à une ancienne image ou à un ancien
  calcul ne peut plus publier son résultat, ses dots ou son erreur après un
  changement de source/paramètres.
- **Textures GPU GUI** : les textures de preview ne sont plus détruites et
  recréées à chaque frame ; elles sont mises à jour uniquement lorsqu'un buffer
  change réellement.
- **Export transparent CLI** : JPEG/BMP sont maintenant composités sur le fond
  au lieu de recevoir directement un buffer RGBA incompatible.
- **SVG défensif** : dimensions, paramètres et coordonnées/rayons non finis ou
  invalides sont rejetés avant génération.
- **Density map** : la mémoire des tables intégrales a été réduite en utilisant
  deux tables `f64` de luminance au lieu de six tables RGB, tout en conservant
  une précision stable sur les grandes images.
- **Audit sécurité** : la chaîne AccessKit responsable des vulnérabilités
  `quick-xml` a été retirée des features `eframe`; les dépendances de codecs
  inutiles par défaut ont également été désactivées.
- **Angle de trame `--grid-angle`** : rotation de la grille (algorithme Grid)
  autour du centre de l'image. Effet "halftone screen" expérimental.
- **Validation uniforme** : `apply()` valide désormais aussi les paramètres des
  algorithmes Grid et Quadtree ; les fréquences et rayons halftone invalides,
  non finis ou négatifs sont rejetés.
- **Density map mise en cache** : une map de longueur incorrecte retourne une
  erreur au lieu de provoquer un panic.
- **Dithering RGBA** : le chemin transparent applique maintenant la même
  quantification Floyd-Steinberg que le chemin RGB, tout en conservant l'alpha.
- **Sauvegardes atomiques** : les fichiers temporaires sont supprimés en cas
  d'erreur et le remplacement de fichiers existants est géré pour Windows.
- **Batch CLI** : le traitement continue après une erreur, mais retourne
  désormais un code d'échec si au moins un fichier n'a pas pu être traité.
- **Documentation des limites** : les plafonds de ressources et la portée des
  paramètres par algorithme sont documentés dans les README et l'architecture.

### Modifié

- **`Algorithm` dérive `clap::ValueEnum`** directement — suppression du
  wrapper `AlgoArg` et des redondances conversion `From<AlgoArg>`.
- **`Args::to_filter_params()`** extraite : la construction `FilterParams`
  depuis les flags CLI est centralisée (main.rs ne fait plus que router).
  La GUI reste inchangée (elle construit `DotShape` directement depuis sliders).

### Documentation

- **`CHANGELOG.md`** au format [Keep a Changelog](https://keepachangelog.com).

### Interface graphique

- **Undo / Redo** (`Ctrl+Z` / `Ctrl+Y` ou `Ctrl+Shift+Z`) : pile d'états
  `FilterParams` avec debounce sur drag. 50 entrées max (FIFO).
- Affichage de l'estimation mémoire de travail et avertissement lors d'un
  redimensionnement automatique d'image.
- Les rayons minimum et maximum sont manipulés dans un contrôle couplé avec
  bornes dynamiques.
- `FilterParams: PartialEq` dérivé pour détecter changements réels.
- Indicateur visuel "Annulé." / "Refait." dans la barre de statut.

### Sauvegarde GUI multi-format

- Outlet dialog étendu : PNG / JPEG / WebP / BMP / TIFF.
- PNG/WebP/TIFF préservent l'alpha ; JPEG/BMP compositent sur `bg_color`
  si `--bg transparent`.

### Affichage GUI transparent

- Preview RGBA composite sur damier 8×8 (gris clair/gris foncé, effet Photoshop)
  via `rgba_to_color_image_checker`. La zone transparente est visible entre dots.

### Accélération GPU

- Feature optionnelle `gpu` avec compute shader WGSL pour calculer la variance
  locale de la density map en parallèle par pixel.
- Activation explicite : `cargo build --features gpu` puis `POINTIMG_GPU=1`.
- Fallback CPU/SAT automatique si l'option runtime est absente ou si aucun
  adapter/device GPU n'est disponible.
- Tests GPU exécutables sur une machine équipée :
  `POINTIMG_GPU=1 cargo test --features gpu`.

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
