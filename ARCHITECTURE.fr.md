# pointimg — Documentation technique

Filtre pointilliste qui décompose une image en points de couleur unie de tailles variables.  
Chaque point prend la couleur moyenne de sa zone et son rayon est modulé par la luminance
et la variance locale.

---

## Sommaire

1. [Structure du projet](#1-structure-du-projet)
2. [Lancement](#2-lancement)
3. [Paramètres (`FilterParams`)](#3-paramètres-filterparams)
4. [Pipeline de traitement](#4-pipeline-de-traitement)
5. [Carte de densité](#5-carte-de-densité)
6. [Calcul du rayon (`radius_for_dot`)](#6-calcul-du-rayon-radius_for_dot)
7. [Algorithmes de placement](#7-algorithmes-de-placement)
8. [Accélération spatiale (`SeedGrid`)](#8-accélération-spatiale-seedgrid)
9. [Rendu (ordre de dessin)](#9-rendu-ordre-de-dessin)
10. [Architecture GUI](#10-architecture-gui)
11. [Export SVG](#11-export-svg)
12. [Outils de qualité](#12-outils-de-qualité)

---

## 1. Structure du projet

```
pointimg/
├── Cargo.toml          — dépendances, deux binaires + une lib
├── src/
│   ├── lib.rs          — racine de la crate lib ; expose `pub mod filter`
│   ├── filter.rs       — toute la logique : algorithmes, rendu, helpers
│   ├── main.rs         — binaire CLI (`pointimg`)
│   └── gui/
│       └── main.rs     — binaire GUI (`pointimg-gui`)
└── assets/             — images de test et sorties
```

| Crate | Rôle |
|---|---|
| `image 0.25` | Chargement / sauvegarde / manipulation de `RgbImage` |
| `clap 4` | Parsing des arguments CLI |
| `rayon 1` | Itération parallèle (carte de densité) |
| `eframe 0.31` | Framework egui (backend wgpu) — *feature-gated* `gui` |
| `egui 0.31` | Widgets GUI en mode immédiat — *feature-gated* `gui` |
| `wgpu 24` | Backend GPU (Vulkan + GL en fallback) — *feature-gated* `gui` |
| `rfd 0.15` | Boîtes de dialogue fichier natives — *feature-gated* `gui` |

> **Feature-gating :** les dépendances GUI sont derrière la feature `gui`
> (activée par défaut). Compiler en CLI-only : `cargo build --no-default-features`.

---

## 2. Lancement

### CLI

```bash
# Voronoï, 1500 points, fond blanc
pointimg -i photo.jpg -o result.png --algorithm voronoi --num-points 1500

# Grille 60 colonnes, fond noir
pointimg -i photo.jpg -o result.png --algorithm grid --cols 60 --bg black

# Fond custom en hexadécimal
pointimg -i photo.jpg -o result.png --bg "#1a1a2e"

# Quadtree, haute sensibilité variance
pointimg -i photo.jpg -o result.png --algorithm quadtree --variance-sensitivity 0.9

# Forme carrée
pointimg -i photo.jpg -o result.png --shape square

# Polygone hexagonal
pointimg -i photo.jpg -o result.png --shape polygon --polygon-sides 6

# Ellipse avec rotation
pointimg -i photo.jpg -o result.png --shape ellipse --ellipse-aspect 2.0 --ellipse-angle 45
```

**Tous les flags :**

| Flag | Court | Défaut | Description |
|---|---|---|---|
| `--input` | `-i` | requis | Image source (JPEG, PNG, RGBA supporté) |
| `--output` | `-o` | `output.png` | Image de sortie PNG |
| `--algorithm` | `-a` | `voronoi` | `grid` \| `kmeans` \| `voronoi` \| `quadtree` |
| `--num-points` | `-n` | `800` | Nombre de points (kmeans/voronoi/quadtree) |
| `--cols` | `-c` | `80` | Colonnes de grille (grid uniquement) |
| `--min-radius` | | `0.003` | Rayon min = fraction de `min(W,H)` |
| `--max-radius` | | `0.06` | Rayon max = fraction de `min(W,H)` |
| `--bg` | `-b` | `white` | `white`, `black`, ou `#rrggbb` |
| `--shape` | | `circle` | `circle` \| `square` \| `ellipse` \| `polygon` |
| `--ellipse-aspect` | | `1.5` | Ratio largeur/hauteur (ellipse) |
| `--ellipse-angle` | | `0.0` | Angle de rotation en degrés (ellipse) |
| `--polygon-sides` | | `6` | Nombre de côtés (polygon, 3-12) |
| `--iterations` | | `10` | Itérations Lloyd / k-means |
| `--variance-sensitivity` | | `0.7` | Force de redistribution par variance `[0,1]` |
| `--max-boost` | | `2.5` | Multiplicateur max dans les zones uniformes |
| `--seed` | | *(aléatoire)* | Seed RNG pour reproduction exacte |
| `--palette` | | *(désactivé)* | Nombre de couleurs dans la palette réduite |
| `--svg` | | *(désactivé)* | Exporter aussi en SVG (même chemin, extension `.svg`) |

### GUI

```bash
pointimg-gui
```

Panneau gauche : tous les paramètres avec sliders + sélecteur d'algorithme.  
Panneau droit : aperçu source / résultat côte à côte.  
Auto-recalcul à chaque changement de paramètre.

---

## 3. Paramètres (`FilterParams`)

```rust
pub struct FilterParams {
    pub algorithm: Algorithm,          // Voronoi par défaut
    pub num_points: usize,             // 800
    pub cols: u32,                     // 80 (Grid uniquement)
    pub min_radius_ratio: f32,         // 0.003
    pub max_radius_ratio: f32,         // 0.06
    pub bg_color: [u8; 3],             // [255,255,255]
    pub iterations: usize,             // 10
    pub variance_sensitivity: f32,     // 0.7
    pub max_boost: f32,                // 2.5
    pub rng_seed: Option<u64>,         // None = horloge système
    pub palette_size: Option<usize>,   // None = toutes les couleurs
    pub dot_shape: DotShape,           // Circle par défaut
}
```

`is_bg_light()` — méthode dérivée de `bg_color` (luminance perceptuelle > 127.5).

**`min_radius_ratio` / `max_radius_ratio`** : fractions de `min(largeur, hauteur)` de l'image.
Sur une image 800×600, `0.003` → 1.8 px et `0.06` → 36 px.

**`variance_sensitivity`** affecte trois endroits :
- La carte de densité (blend entre uniforme et variance pure)
- L'exposant des poids Lloyd (`power = 1 + sensitivity × 3`)
- Le boost de rayon dans les zones plates

**`max_boost`** plafonne le multiplicateur de rayon dans les zones uniformes.
`1.0` = pas de boost. `2.5` = un point en zone totalement plate peut faire 2.5× `r_max`.

---

## 4. Pipeline de traitement

```
src: DynamicImage (JPEG, PNG, RGBA…)
      │
      ▼ flatten_to_rgb()  (composition alpha sur bg_color si RGBA)
      │
      ▼
compute_density_map()   →   density: Vec<f32>   (un float par pixel, parallèle rayon)
      │
      ├─ Grid      → dots_grid()
      ├─ Kmeans    → dots_kmeans_progressive()   (apply_with_progress → callback par itération)
      ├─ Voronoi   → dots_voronoi_progressive()  (apply_with_progress → callback par itération)
      └─ Quadtree  → dots_quadtree()
                              │
                              ▼ quantize_dots() (optionnel, palette_size Some(k))
                              │
                              ▼
                          render()   →   dst: RgbImage
                              │
                              ▼ render_svg_from_dots() (optionnel, export SVG)
```

**`apply_with_progress`** retourne `(RgbImage, Vec<Dot>)` — les dots correspondent
exactement à l'image rendue, ce qui élimine le double-calcul et garantit la cohérence
PNG/SVG. Pour Voronoï et K-means, la dernière itération de la boucle progressive
retourne directement ses dots sans re-rendre l'image (élimination du double rendu).

**Chemin SVG-only (`compute_dots`)** : quand seuls les dots sont nécessaires (export SVG
sans PNG), la fonction `compute_dots()` calcule les dots sans rendre l'image PNG.
En interne, elle dispatche vers `compute_dots_voronoi()` ou `compute_dots_kmeans()`
qui exécutent les itérations Lloyd/K-means et retournent les dots directement.

```
compute_dots()
  ├─ Grid / Quadtree  → apply_with_progress() (rendu classique, dots extraits)
  ├─ Voronoi          → compute_dots_voronoi() (itérations Lloyd sans rendu PNG)
  └─ Kmeans           → compute_dots_kmeans()  (itérations K-means sans rendu PNG)
```

---

## 5. Carte de densité

**Fichier :** `filter.rs` — `compute_density_map()`

**But :** associer à chaque pixel une valeur `[0,1]` indiquant son niveau de détail.
Valeur proche de `1` = zone texturée/contrastée. Proche de `0` = zone plate/uniforme.

**Algorithme (Summed-Area Table) :**

Le calcul utilise une **table d'aires sommées** (SAT) pour obtenir les statistiques
du voisinage en O(1) par pixel, au lieu du O(81) de l'approche naïve par boucle 9×9.

**Phase 1 — Construction des SAT :**

Six tables à prefix-sum de dimensions `(w+1)×(h+1)` sont construites (zero-padded) :
- `sum_r, sum_g, sum_b` — somme des valeurs de canal
- `sq_r, sq_g, sq_b` — somme des carrés (pour le calcul de variance)

Chaque table est remplie en une passe avec la formule SAT classique :
```
SAT[y][x] = val + SAT[y-1][x] + SAT[y][x-1] − SAT[y-1][x-1]
```

**Phase 2 — Requête par pixel :**

Pour chaque pixel `(px, py)`, la fenêtre 9×9 (rayon 4, bords clampés) est interrogée
en O(1) via 4 lectures dans chaque SAT :
```
sum = SAT[y2][x2] − SAT[y1][x2] − SAT[y2][x1] + SAT[y1][x1]
```

1. Calculer la variance par canal R, G, B :
   ```
   var_c = sum_sq_c/n  −  (sum_c/n)²
   ```
2. Moyenner les trois variances :
   ```
   raw = (var_R + var_G + var_B) / 3
   ```
3. Normaliser sur toute l'image (`max_var = max(tous les raw, 1e-6)`) :
   ```
   norm = sqrt(raw / max_var)       ∈ [0, 1]
   ```
   La racine carrée adoucit la courbe (évite que seuls les pixels de bord extrême
   aient une densité élevée).

   > **Note :** `max_var` est clampé à `1e-6` (et non `1.0`) pour préserver le
   > contraste sur les images quasi-uniformes. Sur une image solide, `max_var=0`
   > → epsilon → `norm=0` → `density=0.3` (avec `sensitivity=0.7`), ce qui est correct.

4. Mélange avec `variance_sensitivity` :
   ```
   density[px] = 1 - sensitivity × (1 - norm)
   ```
   - `sensitivity=0` → tout vaut `1.0` (distribution uniforme, variance ignorée)
   - `sensitivity=1` → `density = norm` (redistribution maximale)

**Complexité :** O(W×H) pour la construction des SAT + O(W×H) pour les requêtes.
**Parallélisme :** requêtes row-major via `rayon::par_iter` (les SAT sont en lecture seule).

---

## 6. Calcul du rayon (`radius_for_dot`)

```rust
fn radius_for_dot(lum: f32, local_density: f32, img_min_side: f32, params: &FilterParams) -> f32
```

**Étape 1 — Conversion en pixels :**
```
r_min = min_radius_ratio × img_min_side
r_max = max_radius_ratio × img_min_side
```

**Étape 2 — Modulation halftone (luminance → taille) :**
```
r_lum = r_max − (r_max − r_min) × lum
```
- Pixel sombre (`lum → 0`) → rayon proche de `r_max` (grand point)
- Pixel clair (`lum → 1`) → rayon proche de `r_min` (petit point)

Reproduit l'effet des trames halftone classiques où les zones sombres ont des
points plus grands.

**Étape 3 — Boost d'uniformité :**
```
uniformity = 1 − local_density          ∈ [0, 1]
boost = 1 + uniformity × variance_sensitivity × (max_boost − 1)
résultat = min(r_lum × boost,  max_boost × r_max)
```
- Zone détaillée (`local_density → 1`) : `boost ≈ 1`, pas d'agrandissement
- Zone plate (`local_density → 0`) : `boost` peut atteindre `max_boost`
- Le `min(…, max_boost × r_max)` empêche les points de devenir illimités

**Cap NN (Voronoï / K-means) :**  
Dans `build_dots_from_seeds`, le rayon est de plus plafonné à
`0.8 × (demi-distance au voisin le plus proche)` pour garantir un espace
visible entre les points même dans les zones creuses.

---

## 7. Algorithmes de placement

### 7.1 Grille (`Grid`)

**Complexité :** O(W×H)

1. Diviser l'image en cellules carrées de taille `cell = W / cols`.
2. Pour chaque cellule : calculer la couleur moyenne et la densité moyenne.
3. Émettre un point au centre de la cellule.
4. Rayon plafonné à `cell/2 × 0.8` (80% de la demi-cellule).

**Caractéristique :** distribution spatiale fixe et régulière. Seul le rayon varie.
Rapide, utile comme référence.

---

### 7.2 K-means spatial (`Kmeans`)

**Complexité :** O(iterations × W×H×k) — lent pour k > 500

1. **Initialisation :** `k` graines par `importance_sample` (biaisé vers les zones détaillées).
   Les graines sont placées avec un **jitter sub-pixel** aléatoire de ±0.5 pixel
   pour éviter le clustering sur les centres des pixels.
2. Représenter chaque pixel comme un vecteur 5D normalisé `[x/W, y/H, r/255, g/255, b/255]`.
3. **Itérations :**
   - Assigner chaque pixel au centre le plus proche (distance euclidienne 5D).
   - Recalculer chaque centre comme moyenne de ses pixels assignés.
4. Émettre un point par centre survivant.

**Arrêt anticipé (convergence) :** si le déplacement maximum de tous les centres
est inférieur à 0.5 pixel entre deux itérations, la boucle s'arrête prématurément.
Cela évite les itérations inutiles quand la convergence est déjà atteinte.

**Élimination du double rendu :** à la dernière itération, les dots sont construits
et retournés directement sans re-rendre l'image complète.

**Caractéristique :** regroupe à la fois par proximité spatiale et similarité colorimétrique.
Peut produire des clusters non-convexes. Pas de cap NN appliqué.

---

### 7.3 Voronoï / Lloyd pondéré (`Voronoi`) ← algorithme par défaut

**Complexité :** O(iterations × W×H) grâce à `SeedGrid` — voir §8

1. **Initialisation :** `k` graines par `importance_sample`.
   Les graines sont placées avec un **jitter sub-pixel** aléatoire de ±0.5 pixel
   pour réduire le clustering sur les centres des pixels.
2. **Poids Lloyd :** `w[pixel] = density[pixel] ^ power`
   avec `power = 1 + variance_sensitivity × 3` ∈ [1, 4].
   Les pixels détaillés attirent plus fortement les graines.
3. **Itérations Lloyd pondérées :**
   - Construire une `SeedGrid` depuis les positions actuelles.
   - Pour chaque pixel, trouver la graine la plus proche via la grille.
   - Accumuler : `sum_x[best] += fx × w`, `sum_y[best] += fy × w`, `sum_w[best] += w`.
   - Mettre à jour chaque graine : `seed = (sum_x/sum_w, sum_y/sum_w)`.
4. **Construction des points :** `build_dots_from_seeds` — couleur moyenne de la cellule
   Voronoï + cap NN sur le rayon.

**Caractéristique :** les graines convergent vers les barycentres pondérés de leurs cellules.
Zones détaillées → beaucoup de petits points. Zones plates → quelques gros points.

**Élimination du double rendu :** à la dernière itération, les dots sont construits
et retournés directement sans re-rendre l'image complète.

---

### 7.4 Quadtree adaptatif (`Quadtree`)

**Complexité :** O(W×H×log(max_depth)) amortie

**Paramètres internes :**
```
min_cell  = max(sqrt(W×H / num_points) / 2, 2)   pixels
threshold = 800 × (1 − variance_sensitivity × 0.8)
```

**Subdivision récursive (`subdivide`) :**

```
subdivide(cellule [x,y,w,h]):
  si w < 2 ou h < 2 → stop
  calculer couleur_moyenne et variance de la cellule
  si variance < threshold  OU  taille ≤ min_cell:
      émettre un point au centre
      local_density = min(min(w,h) / img_min, 1.0)
      rayon = radius_for_dot(lum, local_density, img_min, params)
  sinon:
      subdiviser en 4 quarts et récursion
```

La `local_density` est déduite de la taille relative de la cellule :
une grande cellule (zone plate) → `local_density` faible → grand point.

**Caractéristique :** pas de carte de densité explicite. S'adapte directement au contraste
local. Zones plates → une cellule, un gros point. Zones détaillées → cellules minuscules,
petits points denses.

---

## 8. Accélération spatiale (`SeedGrid`)

**Problème résolu :** la recherche naïve du seed le plus proche est O(k) par pixel.
Avec 1500 seeds et une image 800×600, c'est 720 000 × 1500 = **1.08 milliard** de comparaisons
par itération Lloyd.

**Principe :** grille de hachage 2D. Chaque cellule contient la liste des seeds qui y tombent.
La recherche se fait uniquement dans les cellules voisines.

**Construction :**
```
cell_area = (W × H / k) × 4         ≈ 4 seeds par cellule
cell_size = sqrt(cell_area)
cols = ceil(W / cell_size)
rows = ceil(H / cell_size)
```
Chaque seed est inséré dans la cellule correspondant à ses coordonnées.

**Requête nearest(`fx, fy`) :**

Expansion en anneaux concentriques autour de la cellule de `(fx, fy)` :
```
pour radius = 0, 1, 2, ... :
    si distance_minimale_possible(radius)² > best_dist_so_far → stop
    examiner uniquement le bord de l'anneau (pas l'intérieur déjà traité)
    pour chaque seed dans les cellules du bord :
        calculer distance euclidienne, mettre à jour best
```

La distance minimale possible est calculée en tenant compte de la position du point
à l'intérieur de sa cellule : `min_possible = (radius - 1) × cell_size`, clampé à 0
pour `radius ≤ 1` (les cellules adjacentes peuvent toujours contenir un seed plus
proche que le meilleur actuel).

L'arrêt anticipé (`min_possible² > best`) fait que dans le cas moyen,
seules les cellules du voisinage 3×3 sont examinées (~4–8 seeds au lieu de k).

**Complexité effective :** O(pixels × seeds_per_cell) ≈ O(W×H) par itération.

---

## 9. Rendu (ordre de dessin)

```rust
fn render(src: &RgbImage, dots: &[Dot], params: &FilterParams) -> RgbImage
```

Les fonctions `render()` et `render_svg_from_dots()` prennent les dots par
**référence partagée** (`&[Dot]`) et ne les clonent pas — le tri pour l'ordre de
dessin (du plus grand au plus petit) est effectué sur une copie locale du slice.

1. Créer un canvas vide de la couleur de fond.
2. **Trier les points par rayon décroissant** (`sort_unstable_by`).
3. Dessiner du plus grand au plus petit (algorithme du peintre) :
   - Les grands points occupent l'arrière-plan (zones uniformes).
   - Les petits points de détail se superposent au premier plan.
4. Chaque point est dessiné selon `params.dot_shape` (implémentation custom,
   pas de `imageproc`) :

```rust
pub enum DotShape {
    Circle,                                 // Disque plein (défaut)
    Square,                                 // Carré, côté = 2×rayon
    Ellipse { aspect: f32, angle_deg: f32 },// Ellipse avec ratio + rotation
    RegularPolygon { sides: u8 },           // Polygone régulier (3-12 côtés, debug_assert sides≥3)
}
```

```
struct Dot {
    x, y   : f32    // centre en pixels
    color  : [u8;3] // couleur RGB moyenne de la zone
    radius : f32    // rayon en pixels
}
```

---

## 10. Architecture GUI

**Modèle de threading :**

```
Thread principal (egui)               Thread worker
─────────────────────────             ──────────────────────────────────────
App::update() [chaque frame]          filter::apply_with_progress(…, cb)
  ├─ drag & drop → load_image()           │  cb(iter, total, &img) par itération
  ├─ dessiner les widgets                 │  → progress Arc<Mutex<(usize,usize)>>
  ├─ barre de progression Lloyd           │  → result Arc<Mutex<Option<RgbImage>>>
  ├─ temps de calcul (Instant)            │  → ctx.request_repaint() (repaint immédiat)
  ├─ lire computing (AtomicBool)          │  → retourne (RgbImage, Vec<Dot>)
  ├─ si résultat prêt :              ◄─── computing=false, result=Some(img), last_dots=Some(dots)
  │    upload texture egui
  └─ si params changés :
       trigger_compute()
         → cancel=true (annule calcul précédent)
         → wait computing=false
         → start_compute(ctx) (spawn nouveau thread)
```

**Partage inter-thread :**

| Arc | Type | Rôle |
|---|---|---|
| `result` | `Arc<Mutex<Option<RgbImage>>>` | Image produite par le worker |
| `last_dots` | `Arc<Mutex<Option<Vec<Dot>>>>` | Dots du dernier calcul (export SVG) |
| `computing` | `Arc<AtomicBool>` | Flag "calcul en cours" |
| `cancel` | `Arc<AtomicBool>` | Demande d'annulation (vérifié entre itérations) |
| `progress` | `Arc<Mutex<(usize, usize)>>` | (iter_actuelle, iter_totale) pour la barre |
| `compute_error` | `Arc<Mutex<Option<String>>>` | Erreur du thread de calcul |

Tous les `Mutex::lock()` utilisent `unwrap_or_else(|e| e.into_inner())` pour récupérer
le contenu même si le worker a paniqué (poison-safe).

**ViewMode (comparaison avant/après) :**

```rust
enum ViewMode { Side, ResultOnly, SourceOnly, DensityMap }
```

- `Side` : source et résultat côte à côte (divise l'espace disponible en deux)
- `ResultOnly` : résultat plein écran
- `SourceOnly` : source plein écran
- `DensityMap` : aperçu de la density map

**Cache density map :** `src_rgb: Option<RgbImage>` est calculé une seule fois
au chargement (composition alpha incluse), et réutilisé pour chaque recalcul.
`refresh_src_rgb()` est appelé uniquement quand `bg_color` change.
L'aperçu de la density map est **recalculé automatiquement** quand
`variance_sensitivity` change (en plus du chargement et du changement de `bg_color`).

**Preview progressive throttlée :** pour Voronoï et K-means, chaque itération Lloyd
peut publier un résultat intermédiaire via le callback. Le clonage de l'image de
preview est **limité à 100 ms d'intervalle** pour éviter de saturer le thread
principal avec des copies inutiles. La dernière itération est toujours publiée.

**Zoom adaptatif :** le mode « Fit » utilise un flag booléen `zoom_fit` au lieu
de la valeur magique `zoom=0.0`. Le zoom se calcule dynamiquement à chaque frame
en fonction de l'espace disponible. Toute interaction manuelle avec le zoom
(slider, `+`, `-`) désactive `zoom_fit` ; le bouton « Fit » le réactive.

**Raccourcis clavier :**

| Raccourci | Action |
|---|---|
| `Ctrl+O` | Ouvrir une image (dialogue fichier) |
| `Ctrl+S` | Sauvegarder le résultat (dialogue fichier, PNG ou JPEG) |
| `Espace` | Relancer le calcul manuellement |

**Backend GPU :** forcé sur Vulkan (+ GL en fallback) pour éviter le crash
EGL/Wayland avec les drivers NVIDIA sur Wayland :
```rust
WgpuSetup::CreateNew { backends: VULKAN | GL, .. }
```

---

## 11. Export SVG

`render_svg_from_dots(w, h, dots, params) -> Result<String>`

Génère un document SVG autonome avec :
- `<rect>` de fond (couleur `bg_color`)
- Un élément SVG par dot selon `dot_shape` :
  - `Circle` → `<circle cx cy r fill>`
  - `Square` → `<rect x y width height fill>`
  - `Ellipse` → `<ellipse cx cy rx ry transform fill>`
  - `RegularPolygon` → `<polygon points fill>`

La taille du SVG est identique à l'image source en pixels.
Le SVG peut être ouvert dans un navigateur, Inkscape, ou vectorisé davantage.

**API publique :**
```rust
pub fn render_svg_from_dots(w: u32, h: u32, dots: &[Dot], params: &FilterParams) -> Result<String>
pub fn render_svg(src: &RgbImage, params: &FilterParams) -> Result<String>
pub fn render_svg_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<String>
```

---

## 12. Outils de qualité

### Disponibles immédiatement

```bash
# Vérification de types sans compiler
cargo check

# Formatage automatique du code
cargo fmt

# Linter Rust officiel
cargo clippy -- -D warnings

# Tests unitaires (26 tests dans filter.rs)
cargo test --lib

# Générer la documentation Rust
cargo doc --open
```

### Installés

```bash
# Clippy 1.92.0 (depuis rc-buggy)
sudo apt-get install -t rc-buggy rust-clippy=1.92.0+dfsg1-1~exp1

# cargo-audit — détection de vulnérabilités dans les dépendances
sudo apt install cargo-audit
cargo audit
# Note : cargo audit peut échouer si advisory-db contient des entrées CVSS 4.0
# (bug upstream RUSTSEC-2026-0026) — non bloquant
```

### CI suggérée (GitHub Actions)

```yaml
- run: cargo fmt --check
- run: cargo clippy -- -D warnings
- run: cargo test --lib
- run: cargo build --release
```