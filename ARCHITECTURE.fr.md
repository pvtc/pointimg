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
├── Cargo.toml          — dépendances, trois binaires + une lib
├── src/
│   ├── lib.rs          — racine de la crate lib ; expose `filter`, `color`, `frontend`
│   ├── color.rs        — lecture ICC (embarqué/`--input-profile`) et encodage
│   │                     avec profil de sortie (moxcms)
│   ├── frontend.rs     — helpers partagés par les frontends egui et GTK4
│   │                     (formatage, chemins de sortie, écritures atomiques, presets)
│   ├── filter/         — logique séparée : algorithmes, rendu, helpers
│   │   ├── mod.rs      — API publique, ré-exports, tests unitaires
│   │   ├── params.rs   — Algorithm, DotShape, FilterParams, validation, presets
│   │   ├── density.rs  — carte de densité (tables intégrales)
│   │   ├── render.rs   — rendu RGB/RGBA, anti-aliasing, quantization, halftone
│   │   ├── dither.rs   — Floyd-Steinberg
│   │   ├── halftone.rs — HalftoneMode (Cmyk/Dominant), Screening (Am/Fm)
│   │   ├── gamma.rs    — LUTs sRGB ⇄ linéaire
│   │   ├── svg.rs      — export SVG
│   │   ├── seedgrid.rs / sampling.rs / util.rs — accélération et helpers
│   │   ├── gpu.rs      — passe de densité WGSL optionnelle (feature `gpu`)
│   │   └── algorithms/ — grid, kmeans, voronoi, quadtree
│   ├── main.rs         — binaire CLI (`pointimg`)
│   ├── gui/            — interface egui/wgpu (`pointimg-gui`, feature `gui`)
│   └── gtk/            — interface GTK4/libadwaita (`pointimg-gtk`, feature `gtk`)
├── tests/              — tests d'intégration (pipeline, svg, palette, halftone,
│                         reproductibilité, parité GPU)
├── benches/filter.rs   — benchmarks criterion
├── fuzz/               — cibles cargo-fuzz (image, preset) + corpus versionné
└── assets/             — images de test et sorties
```

| Crate             | Rôle                                                             |
| ----------------- | ---------------------------------------------------------------- |
| `image 0.25`      | Chargement / sauvegarde / manipulation de `RgbImage`             |
| `moxcms 0.9`      | Transformations ICC pur Rust vers l'espace de travail sRGB       |
| `clap 4`          | Parsing des arguments CLI                                        |
| `rayon 1`         | Itération parallèle (carte de densité)                           |
| `log` / `env_logger` | Journalisation (`-v`/`-vv`/`-q`, `RUST_LOG`)                  |
| `eframe 0.36`     | Framework egui (backend wgpu) — *feature-gated* `gui`            |
| `egui 0.36`       | Widgets GUI en mode immédiat — *feature-gated* `gui`             |
| `wgpu 30`         | Backend GPU (Vulkan + GL en fallback) — *feature-gated* `gui`/`gpu` |
| `rfd 0.17`        | Boîtes de dialogue fichier natives — *feature-gated* `gui`       |
| `gtk4 0.11`       | Bindings GTK4 — *feature-gated* `gtk`                            |
| `libadwaita 0.9`  | Widgets libadwaita — *feature-gated* `gtk`                       |
| `async-channel 2` | Messagerie worker → UI (GTK) — *feature-gated* `gtk`             |
| `serde 1` / `toml 1` | Presets TOML (`--preset` / `--save-preset`)                   |
| `glob 0.3`        | Expansion glob pour le batch `--input` (`*`, `?`, `[`)           |
| `criterion 0.8`   | Benchmarks — *dev-dependency*                                    |

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

| Flag                     | Court | Défaut        | Description                                           |
| ------------------------ | ----- | ------------- | ----------------------------------------------------- |
| `--input`                | `-i`  | requis        | Image source (JPEG, PNG, RGBA supporté)               |
| `--output`               | `-o`  | `output.png`  | Image de sortie PNG                                   |
| `--algorithm`            | `-a`  | `voronoi`     | `grid` \| `kmeans` \| `voronoi` \| `quadtree`         |
| `--num-points`           | `-n`  | `800`         | Nombre de points (kmeans/voronoi/quadtree)            |
| `--cols`                 | `-c`  | `80`          | Colonnes de grille (grid uniquement)                  |
| `--min-radius`           |       | `0.003`       | Rayon min = fraction de `min(W,H)`                    |
| `--max-radius`           |       | `0.06`        | Rayon max = fraction de `min(W,H)`                    |
| `--bg`                   | `-b`  | `white`       | `white`, `black`, ou `#rrggbb`                        |
| `--shape`                |       | `circle`      | `circle` \| `square` \| `ellipse` \| `polygon`        |
| `--ellipse-aspect`       |       | `1.5`         | Ratio largeur/hauteur (ellipse)                       |
| `--ellipse-angle`        |       | `0.0`         | Angle de rotation en degrés (ellipse)                 |
| `--polygon-sides`        |       | `6`           | Nombre de côtés (polygon, 3-12)                       |
| `--iterations`           |       | `10`          | Itérations Lloyd / k-means                            |
| `--variance-sensitivity` |       | `0.7`         | Force de redistribution par variance `[0,1]`          |
| `--max-boost`            |       | `2.5`         | Multiplicateur max dans les zones uniformes           |
| `--seed`                 |       | *(aléatoire)* | Seed RNG pour reproduction exacte                     |
| `--palette`              |       | *(désactivé)* | Nombre de couleurs dans la palette réduite            |
| `--svg`                  |       | *(désactivé)* | Exporter aussi en SVG (même chemin, extension `.svg`) |

### Profils colorimétriques

Les profils ICC embarqués sont lus par le décodeur `image` puis transformés vers
sRGB avec `moxcms` avant la composition alpha et le pipeline du filtre. Le CLI
accepte aussi `--input-profile srgb`, `--input-profile display-p3` ou un chemin
 vers un fichier `.icc`. Le CLI peut convertir et embarquer un profil ICC de
 sortie via `--output-profile chemin.icc`; sRGB reste le défaut.

La correction gamma conserve le pipeline RGB8 pour Grid, K-means, Voronoi et
Quadtree, car la simplification en points limite l'intérêt d'une précision plus
élevée. Le chemin Halftone CMJN utilise toutefois des couvertures `f32` après
linéarisation, là où la précision influence directement la trame d'encre.

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
    pub transparent: bool,             // fond RGBA transparent
    pub gamma_correct: bool,           // moyennes en espace linéaire
    pub dithering: bool,               // Floyd-Steinberg
    pub halftone: HalftoneMode,        // Off | Cmyk | Dominant
    pub screening: Screening,          // Am | Fm
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

**Validation et limites :** les paramètres sont contrôlés avant tout calcul.
Les images sont limitées à 65 535 pixels par côté et 8 millions de pixels au
total. Le moteur accepte au plus 100 000 points, 50 000 points pour K-means/Voronoi,
8 192 colonnes, 100 itérations
et une palette de 2 à 256 couleurs. Les valeurs flottantes doivent être finies ;
la fréquence halftone doit être strictement positive.

### Format des presets

Les presets sauvegardés utilisent une enveloppe versionnée :

```toml
preset_version = 1

[params]
algorithm = "Voronoi"
num_points = 800
```

Les anciens presets dont les paramètres sont directement à la racine restent
lisibles. Les versions futures inconnues sont refusées explicitement.

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

Deux tables `f64` à prefix-sum de dimensions `(w+1)×(h+1)` sont construites (zero-padded) :
- `sum_l` — somme de la luminance
- `sq_l` — somme des carrés de luminance (pour le calcul de variance)

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

1. Calculer la variance de luminance :
   ```
   var_l = sum_sq_l/n  −  (sum_l/n)²
   ```
2. Utiliser cette variance comme mesure de détail :
   ```
   raw = var_l
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

```text
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

**Complexité :** O(iterations × W×H×candidats) — la variante spatiale
`nearest_by` de `SeedGrid` rend le nombre de candidats par pixel ≪ k, et
l'arrêt anticipé au point fixe saute les itérations redondantes.

1. **Initialisation :** `k` graines par `importance_sample` (biaisé vers les zones détaillées).
   Les graines sont placées avec un **jitter sub-pixel** aléatoire de ±0.5 pixel
   pour éviter le clustering sur les centres des pixels.
2. Représenter chaque pixel comme un vecteur 5D normalisé `[x/W, y/H, r/255, g/255, b/255]`.
   Les positions/couleurs normalisées sont précalculées une fois (LUTs) au
   lieu d'être re-divisées par pixel à chaque itération.
3. **Itérations :**
   - Assigner chaque pixel au centre le plus proche (distance euclidienne 5D)
     via `SeedGrid::nearest_by` — la distance spatiale² est une borne
     inférieure de la distance totale², donc les anneaux au-delà de `best²`
     sont sautés. Les égalités sont départagées par le plus petit index,
     exactement comme `Iterator::min_by`.
   - Les sommes partielles s'accumulent par **chunks de lignes fixes**
     combinés dans l'ordre des indices → réduction f64 déterministe (plus de
     dépendance au work-stealing, contrairement à l'ancien `fold/reduce`).
   - Recalculer chaque centre comme moyenne de ses pixels assignés.
4. Émettre un point par centre survivant.

**Arrêt anticipé (convergence) :** la boucle s'arrête dès que tous les
centres sont bit-à-bit inchangés (point fixe exact) : toute itération
supplémentaire serait un no-op exact, donc le résultat final est inchangé.

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

**Complexité :** O(W×H) pour construire les tables intégrales + O(1) par
requête de nœud — les **summed-area tables** u64 entières (sommes et sommes
des carrés par canal RGB, même technique que la carte de densité) remplacent
les anciens re-parcours O(surface) par nœud.

**Paramètres internes :**
```
min_cell  = max(sqrt(W×H / num_points) / 2, 2)   pixels
threshold = 800 × (1 − variance_sensitivity × 0.8)
```

**Subdivision récursive (`subdivide`) :**

```
subdivide(cellule [x,y,w,h]):
  si w < 2 ou h < 2 → stop
  calculer couleur_moyenne et variance de la cellule via les SATs
  (expansion exacte en entiers : Σ(p−a)² = Σp² − 2a·Σp + n·a²)
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

**`nearest_by` (variante K-means) :** recherche par anneaux généralisée qui
prend une closure de distance sur les indices des seeds, plus deux facteurs
convertissant les pixels de la grille en unités spatiales de la distance
(ex. coords K-means normalisées). Les égalités sont départagées par le plus
petit index, comme `Iterator::min_by` — bit-à-bit identique au balayage naïf.

---

## 9. Rendu (ordre de dessin)

```text
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
4. **Bandes parallèles (rayon) :** `render`/`render_rgba` découpent le canvas
   en bandes horizontales. Chaque bande applique tous les dots dans l'ordre
   global par rayon en bornant les pixels à ses lignes — résultat identique
   au dessin séquentiel — via `draw_dot_clipped` et une closure d'écriture
   par bande (`blend_rgb_band`/`blend_rgba_band`).
5. Chaque point est dessiné selon `params.dot_shape` (implémentation custom,
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

Deux front-ends partagent la bibliothèque (`src/frontend.rs` pour le formatage,
les écritures atomiques, l'export et les presets) :

| Binaire     | Bibliothèque      | Rôle                                             |
| ----------- | ----------------- | ------------------------------------------------ |
| `pointimg-gui` | `eframe`/`egui` | rendu immédiat, style egui par défaut           |
| `pointimg-gtk` | `gtk4`/`libadwaita` | widgets natifs, `adw::PreferencesPage`        |

### 10.1 Décisions d'interface (communes aux deux)

- **Actions dans l'en-tête** : ouvrir, dossier, presets, recalculer,
  annuler/rétablir, exporter, copier. Le panneau ne contient que des
  *réglages* ; plus rien d'important sous la ligne de flottaison.
- **Comparaison avant/après** (`ViewMode::Compare`, mode par défaut) : la source
  et le résultat sont dessinés dans la même zone, de part et d'autre d'un
  curseur. GTK : `gtk::DrawingArea` + surfaces cairo découpées au curseur
  (`src/gtk/compare.rs`). egui : `Painter::image` avec deux `clip_rect`.
- **Aperçu rapide + finalisation** : `State.quality` ∈ {`Half`, `Quarter`,
  `Full`}. Tant qu'il n'est pas `Full`, le worker reçoit une image
  sous-échantillonnée (`compute_source()`) ; après ~900 ms sans modification,
  une passe `final_pass` en pleine résolution est déclenchée. Le résultat
  n'est `result_is_final` qu'à l'arrivée de cette passe : l'export et la
  sauvegarde SVG restent désactivés tant qu'il est obsolète.
- **Density map indexée par l'échelle** : `State.density_scale` memorize
  l'échelle du cache ; une density map calculée en ½ n'est pas réutilisée par
  une passe pleine résolution (les tailles ne correspondent pas).
- **Presets intégrés** : `filter::builtin_presets()` est dans la bibliothèque
  (testée : chaque preset passe `validate_params` et l'aller-retour TOML).
  Les deux interfaces proposent la liste en plus des fichiers TOML.
- **Saisie directe** : GTK utilise `adw::SpinRow` (glissière *et* saisie) avec
  les bornes de `validate_params` ; egui utilise `DragValue` avec les mêmes
  plages. Les curseurs s'arrêtent donc exactement là où le moteur refuse.
- **Molette dans le panneau** : GTK — un `EventControllerScroll` en phase
  *capture* redirige la molette vers le `ScrolledWindow` parent (avec Ctrl/Shift,
  le cran ajuste le réglage). Sans cela, faire défiler le panneau modifiait
  silencieusement les valeurs *et* relançait un calcul par cran.
- **Vues synchronisées** : le défilement d'un panneau d'aperçu est répercuté
  sur les autres (position relative, pas valeur absolue).
- **Zoom** : le mode « Ajuster » calcule le facteur depuis le viewport et
  l'affiche tel quel (le widget de zoom est insensible dans ce mode) ; tout
  changement de zoom conserve le centre du viewport au lieu de sauter vers le
  coin haut-gauche.
- **État vide** : `adw::StatusPage` / centre egui avec « Ouvrir une image »,
  « Ouvrir un dossier » et « Utiliser un exemple » (si `assets/examples/` est
  présent à côté du binaire).

### 10.2 egui (`src/gui/`)

```
Thread principal (egui)               Thread worker
─────────────────────────             ──────────────────────────────────────
eframe::App::ui() [chaque frame]      filter::apply_with_progress(…, cb)
  ├─ résultats des workers (poll)         │  cb(iter, total, &img) par itération
  ├─ raccourcis / glisser-déposer         │  → progress Arc<Mutex<(usize,usize)>>
  ├─ annulation + debounce               │  → result Arc<Mutex<Option<RgbaImage>>>
  ├─ barre de progression Lloyd           │  → result_is_final Arc<AtomicBool>
  ├─ upload des textures egui             │  → retourne (RgbImage, Vec<Dot>)
  ├─ finalisation après inactivité   ◄─── computing=false, result=Some(img)
  └─ si params changés :
       on_params_changed() → debounce 300 ms
       → trigger_compute() → cancel puis start_compute()
```

Panneaux : `egui::Panel::top` (actions), `egui::Panel::bottom` (état),
`egui::Panel::left` (réglages, dans un `ScrollArea`), `egui::CentralPanel`
(apercus). **L'ordre d'appel compte** : le panneau du bas doit être réservé
avant celui de gauche, sinon le panneau de gauche déborde dessus.

### 10.3 GTK4 / libadwaita (`src/gtk/`)

```
Thread principal GTK                  Threads workers
─────────────────────────             ──────────────────────────────────────
State (Rc<RefCell<..>>) + Widgets    filter::apply_with_progress(…, publish)
  ├─ actions win.* (SimpleActionGroup)   │  publish → Msg::Progress / Preview
  ├─ signaux widgets (wire.rs)           │  → canal async-channel
  ├─ Annuler le calcul (AtomicBool)  ◄─── Msg::Done { image, dots, elapsed_ms,
  ├─ zoom / vues (preview.rs)                 full_resolution }
  └─ spawn_local : boucle sur le canal   Msg::Failed / Msg::Density
```

- `State` est fondé sur un **compteur de génération** : un worker obsolète ne
  publie jamais (`is_current`), que ce soit pour un résultat, une progression
  ou une erreur.
- Toute écriture d'état par un callback GTK relâche son `borrow` **avant** de
  toucher un widget : les setters GTK émettent des signaux synchrones dont les
  handlers rappellent `state.borrow()` (panic « already mutably borrowed »).
- Un drapeau `updating` neutralise les handlers pendant les mises à jour
  programmatiques (undo, preset, preset de fichier).
- Les actions `win.*` sont déclarées dans un `SimpleActionGroup` inséré sur la
  fenêtre ; les accélérateurs passent par un `ShortcutController` greffé à la
  fenêtre (`gtk_shortcut_action_parse_string` attend `action(win.nom)`), les
  raccourcis non-`Ctrl` par un `EventControllerKey`.

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
```text
pub fn render_svg_from_dots(w: u32, h: u32, dots: &[Dot], params: &FilterParams) -> Result<String>
pub fn render_svg(src: &RgbImage, params: &FilterParams) -> Result<String>
pub fn render_svg_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<String>
```

---

## 12. Outils de qualité

### Disponibles immédiatement

```bash
# Vérification de types sans compiler (gtk exclu : exige GTK4/libadwaita)
cargo check --features gui,gpu,avif
cargo check --no-default-features          # CLI seul

# Formatage automatique du code
cargo fmt

# Linter Rust officiel
cargo clippy --features gui,gpu,avif --all-targets -- -D warnings
cargo clippy --no-default-features --all-targets -- -D warnings

# Tests unitaires (src/filter/* + src/main.rs + src/color.rs)
cargo test --lib

# Tests d'intégration (tests/*.rs)
cargo test --features gui,gpu,avif
cargo test --no-default-features            # build CLI seul

# Benchmarks (criterion)
cargo bench

# Générer la documentation Rust
cargo doc --no-deps --features gui,gpu,avif
```

### Installés

```bash
# cargo-audit — détection de vulnérabilités dans les dépendances
cargo install cargo-audit
cargo audit
# Note : la cible optionnelle `avif` tire `paste` (RUSTSEC-2024-0436, alerte de
# maintenance) documentée dans audit.toml — non bloquant.
```

### CI (GitHub Actions)

Le pipeline complet vit dans `.github/workflows/ci.yml` et tourne à chaque
push/PR sur `main` :

- **fmt** — `cargo fmt --check`
- **clippy** — matrice `ubuntu/macos/windows`,
  `cargo clippy --features gui,gpu,avif --all-targets -- -D warnings` +
  `--no-default-features --all-targets` (couverture CLI seule)
- **test** — matrice `ubuntu/macos/windows`,
  `cargo test --features gui,gpu,avif --locked` + `--no-default-features --locked`
- **gpu** — adaptateur logiciel lavapipe, parité CPU/GPU de la densité
- **msrv** — `cargo +1.95 check --features gui,gpu,avif --locked` (et `stable`)
- **build** — `cargo build --release --features gui,gpu,avif`
- **gtk** — Linux uniquement, installe `libgtk-4-dev`/`libadwaita-1-dev`,
  clippy + build release de `pointimg-gtk`
- **docs** — `cargo doc --no-deps --features gui,gpu,avif` avec `--cfg docsrs`
- **audit** — `cargo audit` (+ workflow planifié)
- **fuzz-build** — smoke tests nightly `cargo fuzz` (100 runs par cible)

`gtk` est volontairement exclu de `--all-features` partout : GTK4 et
libadwaita ne sont pas disponibles sur tous les runners ni sur docs.rs.
