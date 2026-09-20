# Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Corrigé

- **Conflit d'ID egui** : en vue côte-à-côte (Source/Résultat), les deux
  `ScrollArea` de la zone image partageaient le même salt par défaut et
  recevaient le même ID persisté — egui affichait une erreur rouge. Chaque
  panneau utilise désormais son label comme `id_salt`. (`src/gui/ui.rs`)
- **`--jobs 0` honoré en batch** : le flag annonçait « 0 = tous les CPUs » mais
  le pool était construit avec `num_threads(max(1))`, soit un seul thread
  (séquentiel déguisé). `num_threads(0)` laisse désormais rayon choisir tous
  les CPUs disponibles. (`src/main.rs`)
- **Clippy `--all-targets`** : `tests/gpu_density.rs` utilisait
  `(v / 2).min(255)`, inutile, ce qui faisait échouer `-D warnings` dès que les
  tests étaient lintés. (`tests/gpu_density.rs`)
- **`--algorithm halftone` sans `--halftone`** : le CLI renvoyait une erreur
  alors que son commentaire et sa documentation annonçaient un défaut `cmyk`.
  Le mode `cmyk` est désormais activé automatiquement. (`src/main.rs`)
- **Compositing alpha arrondi** : `flatten_to_rgb` tronquait (`as u8`) au lieu
  d'arrondir, contrairement au chemin CLI/GUI — léger biais vers le sombre et
  incohérence entre frontends. (`src/filter/util.rs`)
- **Export SVG GUI** : l'extension est maintenant forcée à `.svg` (un fichier
  choisi en `.png` recevait auparavant du contenu SVG). (`src/frontend.rs`)
- **Sauvegardes GTK atomiques** : le frontend GTK4 écrit désormais via un
  fichier temporaire + remplacement atomique, comme egui, et arrondit le
  compositing alpha. (`src/gtk/files.rs`)
- **Halftone AM** : suppression de `height_or_h` (une identité inutile) et du
  calcul de rotation inverse aussitôt jeté ; un commentaire contenait des
  caractères chinois. (`src/filter/halftone.rs`)
- **Estimation mémoire halftone** : `estimate_memory_bytes` ignorait les cartes
  de couverture (jusqu'à ~1 Go en mode dominant), ce qui sous-évaluait fortement
  la mémoire annoncée par la GUI. (`src/filter/util.rs`)
- **Sur-allocation des buffers ICC** : `transform_rgb`/`transform_rgba`
  réservaient `source.len() * 3` / `* 4` éléments `f32` alors que `source.len()`
  est déjà le nombre de composantes — soit 3 à 4 fois la mémoire nécessaire
  (~500 Mo inutiles pour une image RGBA au plafond). (`src/color.rs`)
- **Faux positif « profil converti »** : avec `--input-profile auto` (défaut)
  et aucune profile embarqué, le CLI/GUI annonçaient une conversion
  colorimétrique inexistante. (`src/color.rs`)
- **`importance_sample` dégénéré** : quand `num_points ≥ nombre de pixels`, tous
  les points excédentaires s'écrasaient sur le dernier pixel, dégénérant
  Lloyd/k-means. Ils sont désormais répartis sur tous les pixels.
  (`src/filter/sampling.rs`)
- **Panics gamma + halftone** : trois routes publiques atteignaient
  `unreachable!()` — `apply_dynamic` (gamma + halftone), `apply` (gamma +
  halftone **dominant**, seul le CMJN était routé) et
  `apply_with_progress_cached` (halftone sans gamma). Chacune délègue désormais
  au pipeline halftone dédié. (`src/filter/mod.rs`)
- **Bornes halftone manquantes** : `halftone_min_radius_ratio` et
  `halftone_max_dot_ratio` n'avaient pas de plafond ; un preset ou un flag
  arbitrairement grand (`--halftone-max-dot 1e9`) produisait des dots couvrant
  toute l'image → rendu pathologiquement lent. Bornés à 1.0 et 10.0.
  (`src/filter/params.rs`)
- **Clone inutile de la density map (GUI)** : la density map était clonée
  intégralement avant d'être publiée dans l'`Arc` — jusqu'à ~33 Mo par recalcul.
  L'image est désormais construite avant le déplacement. (`src/gui/compute.rs`)
- **`density_to_image` arrondit** au lieu de tronquer, comme les autres
  conversions u8 du projet. (`src/filter/density.rs`)
- **Crash GTK « RefCell already mutably borrowed »** : le bouton zoom `1:1`
  gardait un `state.borrow_mut()` ouvert pendant `set_value`/`set_active`, dont
  les signaux synchrones rappelaient `state.borrow()` → panic (abort). Le
  borrow est désormais relâché avant de toucher les widgets ; les rafraîchisseurs
  de textures ont été durcis de la même façon. (`src/gtk/wire.rs`,
  `src/gtk/preview.rs`)

### Ajouté

- **`filter::estimate_memory_bytes_for(w, h, params)`** : estimation tenant
  compte des cartes de couverture halftone (4 o/px en CMYK, `n` en dominant).
  Utilisée par le CLI et les deux frontends. (`src/filter/util.rs`)
- **Fuzzing image élargi** : la cible `image` exerce désormais le pipeline
  complet (Grid/Kmeans/Voronoi/Quadtree + export SVG) et le halftone dominant FM
  sur une image réduite à 64 px. (`fuzz/fuzz_targets/image.rs`)
- **Tests color management** : la transformation ICC par chunks est comparée
  bit-à-bit à un passage unique (RGB et RGBA) ; test de répartition
  d'`importance_sample` ; test d'arrondi du compositing alpha.
  (`src/color.rs`, `src/filter/sampling.rs`, `src/filter/util.rs`)
- **Tests anti-panic halftone** : `apply_dynamic` gamma+CMJN, `apply` gamma+dominant
  et `apply_with_progress_cached` halftone ; bornes des rayons halftone.
  (`tests/halftone.rs`, `src/filter/mod.rs`)
- **Tests des mappings GTK** : aller-retour `algorithm_index`/`view_index` et
  `shape_index` (fonctions pures jusqu'ici non testées). (`src/gtk/sections.rs`)

- **Module `pointimg::frontend`** : helpers partagés entre les frontends egui et
  GTK4 (formatage, normalisation d'extension, écritures atomiques, export RGBA,
  presets). Supprime les duplications `format_duration`/`format_memory`,
  `with_extension`, `flatten_rgba`, `rgb_to_rgba_opaque`. (`src/frontend.rs`)
- **Tests CLI** : `parse_bg_color`, `parse_halftone`, `parse_screening`,
  `parse_preview_size`, `resolve_output_path` (substitutions et `--svg`) et
  défaut halftone. (`src/main.rs`)
- **Tests de non-régression palette** : la quantification dédupliquée est
  comparée bit-à-bit à l'ancienne implémentation de référence. (`src/filter/render.rs`)
- **Tests du module frontend** : formatage, extensions forcées, compositing,
  round-trip de preset. (`src/frontend.rs`)

- **Interface GTK4 / libadwaita** : nouveau binaire `pointimg-gtk`, derrière la
  feature optionnelle `gtk` (GTK 4.10+, libadwaita 1.4+). Parité fonctionnelle
  avec la GUI egui (5 algorithmes, formes, palette/dithering, fond et
  transparence, gamma, halftone CMYK/dominant AM/FM, density map, zoom/Fit/1:1,
  presets, undo/redo, export PNG/JPEG/WebP/BMP/TIFF/SVG, glisser-déposer,
  raccourcis `Ctrl+O/S/Z/Y` et `Espace`, aperçu côte à côte). Les calculs
  tournent dans un thread dédié avec preview progressive, debounce et
  annulation ; les textures GDK sont rafraîchies sur le thread principal via un
  canal asynchrone. (`src/gtk/*`)
- **CI GTK4** : job dédié installant `libgtk-4-dev` et `libadwaita-1-dev`, avec
  `clippy -- -D warnings` et build release du binaire `pointimg-gtk`.
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
- **MSRV déclarée** : Rust 1.95 (`rust-version` dans `Cargo.toml`).
- **Découpage modulaire** de `src/filter.rs` (2072 lignes) en 10 fichiers :
  `params`, `density`, `render`, `svg`, `seedgrid`, `sampling`, `util`,
  `algorithms/{grid,kmeans,voronoi,quadtree}`, `dither`, `gamma`, `halftone`.
- **Tests d'intégration** (`tests/{pipeline,svg,reproducibility,palette,halftone}.rs`)
  et benches Criterion (`benches/filter.rs`).
- **CI GitHub Actions** réécrite : cache `Swatinem/rust-cache`, matrix
  Ubuntu/macOS/Windows, job `msrv` (1.95 + stable), job `audit`, job `docs`
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
- **Badges README** : MSRV 1.95 + exemples + docs des nouveaux flags (EN/FR).

### Outillage / CI

- **Clippy couvre tous les targets** : la CI lint désormais tests, benches et
  exemples via `--all-targets`, ce qui aurait détecté le lint de
  `tests/gpu_density.rs`.
- **Couverture CLI-only** : les jobs clippy et test exécutent aussi
  `--no-default-features --all-targets`, et le job GTK lance la suite de tests
  avec `--features gtk`. Le build sans dépendances GUI est ainsi protégé.
  (`.github/workflows/ci.yml`)
- **`rust-toolchain.toml`** : canal `stable` avec `rustfmt` et `clippy`
  déclarés, pour un environnement local reproductible (les jobs MSRV
  épinglent toujours `+1.95`).
- **Release GTK4** : le binaire `pointimg-gtk` est construit et publié
  (`libgtk-4-dev`/`libadwaita-1-dev`) aux côtés des autres artefacts. Le job de
  la GUI egui (`pointimg-gui`) n'installe plus `libgtk-3-dev`/`libspeechd-dev`,
  inutiles : `rfd` 0.17 passe par le portail XDG, et le front GTK4 est un
  binaire distinct.
- **Audit** : l'alerte de maintenance `RUSTSEC-2024-0436` (`paste`, tiré par
  l'encodeur AVIF optionnel) est documentée dans `audit.toml`.
- **Corpus de fuzzing** versionné (`fuzz/corpus/`) comme graines de régression.
- **Release reproductible** : tous les builds de `release.yml` utilisent
  `--locked`, et `softprops/action-gh-release` passe en `v2`.
  (`.github/workflows/release.yml`)
- **Fuzz réparé en CI** : le `rust-toolchain.toml` stable de la racine écrasait
  le nightly installé par le job, donc `cargo fuzz` échouait (`-Z` refusé sur
  stable). Un `fuzz/rust-toolchain.toml` (nightly) et `RUSTUP_TOOLCHAIN=nightly`
  dans le job garantissent le bon toolchain. (`.github/workflows/ci.yml`,
  `fuzz/rust-toolchain.toml`)
- **Graine de fuzzing image réelle** : le corpus `image` ne contenait qu'un
  fichier de 4 octets (jamais décodé) ; une petite image PNG valide
  (`fuzz/corpus/image/seed-image-12x12`) fait passer la couverture du target de
  331 à 10 501 branches. (`fuzz/corpus/image/`)

### Modifié

- **CI et docs.rs** : la feature `gtk` est exclue de `--all-features` (GTK4 et
  libadwaita ne sont pas disponibles sur tous les runners ni sur docs.rs). Les
  jobs clippy/test/msrv/build/docs utilisent désormais `--features gui,gpu,avif`
  et `[package.metadata.docs.rs]` liste explicitement ces features.
- **Mise à jour des dépendances** : `egui`/`eframe` 0.31 → 0.36 (trait
  `App::ui` et `egui::Panel` unifié), `wgpu` 24 → 30 (API instance/adapter/
  pipeline), `rfd` 0.15 → 0.17, `moxcms` 0.8 → 0.9, `toml` 0.8 → 1,
  `criterion` 0.5 → 0.8 (`black_box` via `std::hint`), plus les mises à jour
  transitives. Adaptations dans `src/gui/{ui,main,convert}.rs`,
  `src/filter/gpu.rs` et `benches/filter.rs`.
- **Accélération K-means (~15× sur bench, résultat bit-à-bit identique)** :
  recherche du centre le plus proche via `SeedGrid::nearest_by` (borne
  inférieure sur la distance spatiale², départage des égalités par index
  comme `min_by`), LUTs de normalisation précalculées, accumulation f64 en
  chunks fixes déterministes (évite la dépendance au work-stealing), arrêt
  précoce au point fixe exact. (`src/filter/algorithms/kmeans.rs`,
  `src/filter/seedgrid.rs`)
- **Accélération Quadtree** (`src/filter/algorithms/quadtree.rs`) :
  tables intégrales u64 (sommes + sommes de carrés RGB) → requêtes de
  somme/variance O(1) au lieu de re-parcourir les pixels du nœud.
- **Rendu par bandes parallèles (rayon)** : `render`/`render_rgba` dessinent
  les dots par bandes horizontales en préservant l'ordre global par rayon
  — résultat pixel-identique, plusieurs fois plus rapide. (`src/filter/render.rs`)
- **`apply()` sans previews gaspillées** : pour Kmeans/Voronoi, le chemin
  non-progressif calcule les dots directement puis rend une seule fois au
  lieu de jeter un rendu complet à chaque itération. (`src/filter/mod.rs`)
- **Estimation mémoire** à jour pour les tables intégrales du quadtree
  (6 u64/canal-pixel). (`src/filter/util.rs::estimate_memory_bytes`)
- **Quantification de palette accélérée** : les couleurs des dots sont
  dédupliquées une seule fois et le centre le plus proche n'est cherché qu'une
  fois par couleur unique (sortie anticipée sur correspondance exacte). Le
  résultat reste bit-à-bit identique (accumulation f64 dans l'ordre d'origine).
  (`src/filter/render.rs`)
- **Halftone dominant : mémoire ÷4** : les cartes de couverture passent de
  `f32` à `u8` (quantification 1/255, visuellement neutre). Le pic passe
  d'environ 1 Go à ~268 Mo pour `n = 32` sur une image au plafond.
  (`src/filter/halftone.rs`)
- **Halftone dominant : k-means dédupliqué** : les couleurs de pixels sont
  dédupliquées une fois ; le centre le plus proche n'est cherché qu'une fois par
  couleur unique (au lieu d'une fois par pixel et par itération), à résultat
  identique. (`src/filter/halftone.rs`)
- **Export SVG accéléré** : écriture directe via `write!` dans un `String`
  pré-dimensionné (plus de `format!` par dot ni de `Vec<String>` + `join` pour
  les polygones), sortie inchangée. (`src/filter/svg.rs`)
- **Dithering Floyd-Steinberg accéléré** : palette pré-convertie en `f32`,
  recherche manuelle du plus proche avec sortie anticipée sur correspondance
  exacte (mêmes résultats). (`src/filter/dither.rs`)
- **Voronoï refactoré** : la boucle de Lloyd pondérée est extraite dans
  `run_lloyd`, partagée entre le chemin progressif et le chemin direct
  (auparavant ~60 lignes dupliquées). (`src/filter/algorithms/voronoi.rs`)
- **Transformation ICC par chunks** : les buffers `f32` intermédiaires sont
  limités à 64 k pixels par appel au lieu de deux buffers pleine image
  (24–32 o/px) ; sortie identique. (`src/color.rs`)
- **Profil de release** : `codegen-units = 1` (meilleure optimisation
  inter-modules) et `strip = true` (binaires sans symboles). (`Cargo.toml`)
- **Benchmarks complétés** : halftone `dominant-6-am` et estimation mémoire
  tenant compte du halftone. (`benches/filter.rs`)
- **`apply_dynamic` simplifié** : aplatit puis délègue à `apply`, ce qui
  supprime la duplication du chemin gamma et le panic associé.
  (`src/filter/mod.rs`)
- **CLI dédupliqué** : `main.rs` réutilise `pointimg::frontend` pour les
  écritures atomiques, l'aplatissement RGBA et la détection d'alpha
  (suppression de ~90 lignes dupliquées). (`src/main.rs`)
- **`Ctrl+S` cohérent** : la boîte de dialogue propose désormais WebP/BMP/TIFF
  comme le bouton d'export, au lieu de PNG/JPEG seulement. (`src/gui/ui.rs`)

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
  `quick-xml` avait été retirée des features `eframe`; elle est réactivée
  (`accesskit`, support des lecteurs d'écran) maintenant que `quick-xml` 0.41 a
  corrigé les advisories. Les dépendances de codecs inutiles par défaut restent
  désactivées.
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
