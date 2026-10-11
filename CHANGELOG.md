# Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Ajouté

- **Vignettes de presets** : les neuf presets intégrés sont maintenant choisis
  sur leur rendu et non sur leur nom. Chaque vignette est produite avec les
  réglages du preset lui-même, par `cargo run --example preset_thumbnails`, sur
  une image d'exemple — un portrait, seul exemple du dépôt où l'on voit ce
  qu'un preset fait d'un sujet reconnaissable. egui les affiche en grille de trois
  colonnes, GTK dans un `GtkFlowBox` ; la vignette courante est encadrée. Sans
  le dossier `assets/presets/`, les deux interfaces retombent sur une liste de
  noms. (`examples/preset_thumbnails.rs`, `src/frontend.rs`,
  `tests/preset_thumbs.rs`)
- **Images récentes dans egui** : l'état d'accueil propose les dernières images
  ouvertes, conservées dans le fichier de préférences. GTK lit la même
  information dans `GtkRecentManager`. (`src/gui/prefs.rs`, `src/gui/ui.rs`)
- **Sections repliables dans egui** : le panneau de réglages était une colonne
  de trente curseurs à parcourir en entier pour trouver une option. Quatre
  sections (`Algorithme`, `Forme des points`, `Fond et rendu`, `Calcul`) se
  replient, et l'état de repli est mémorisé d'un lancement à l'autre.
  (`src/gui/ui.rs`)
- **Palette de commandes** (`Ctrl+K`) dans egui : vingt actions avec leur
  raccourci affiché, commandes indisponibles grisées mais visibles. L'interface
  n'avait ni menu ni fenêtre de raccourcis — ses dix-sept raccourcis n'étaient
  découvrables qu'au survol, et la carte de densité comme le plein écran
  étaient inatteignables. (`src/gui/palette.rs`)
- **Notifications dans egui** : erreurs et confirmations passent au-dessus de
  l'aperçu (3 s, 8 s pour une erreur). Une erreur de calcul était invisible
  pendant le calcul même, la ligne d'état étant remplacée par la progression.
  (`src/gui/notify.rs`)
- **Images récentes dans GTK** via `GtkRecentManager`, la liste du bureau, sous
  les boutons de l'état d'accueil. (`src/gtk/recent.rs`)
- **Ouverture par la ligne de commande** : `pointimg-gtk photo.jpg` et
  `pointimg-gui photo.jpg`, ainsi qu'un dossier qui active la navigation. GTK
  passe par `HANDLES_OPEN`, donc aussi par l'activation du gestionnaire de
  fichiers. (`src/gtk/main.rs`, `src/gui/main.rs`)
- **Entrées de bureau** (`%F`, donc l'ouverture depuis le gestionnaire de
  fichiers) dans `packaging/`, avec le jeu d'icônes `hicolor`. (`packaging/`)
- **Logo** : trois points primaires cyan, magenta et jaune de tailles
  décroissantes. Le SVG est la source, `packaging/icons/regenerate.sh` en
  dérive toutes les tailles ; le PNG 256 px est embarqué dans le binaire egui,
  qui n'a pas de thème d'icônes. (`assets/logo/`)
- **Tests de plages** : chaque interface déclare ses bornes et vérifie les deux
  extrémités de chaque curseur contre `validate_params`, plus la valeur
  juste au-delà. (`src/gui/ranges.rs`, `src/gtk/ranges.rs`)

- **Chemin sans aperçus pour la CLI** (`filter::apply_with_iteration_counts`) :
  la CLI rendait une image complète à chaque itération de Lloyd, image que son
  callback jetait. Elle n'annonce plus que la progression. **1,21× mesuré** sur
  un aperçu 1000×800 / 10 itérations / 4000 points, sortie identique au bit.
  (`src/filter/mod.rs`, `tests/no_preview_path.rs`)
- **Logique partagée extraite** dans `frontend`, sans toucher aux widgets :
  `random_seed` (la graine tirée par « Varier la graine »), `derived_export_name`
  (cinq copies), `matching_preset_index` (quatre copies), `collect_folder_images`
  et `FolderError`, `downsample`, les bornes et le pas de zoom, et un module
  `msg` de treize messages. Ces éléments étaient écrits une fois par interface,
  chacune pouvant diverger en silence. (`src/frontend.rs`)
- **Damier de transparence défini une seule fois** : la formule du composite
  était écrite trois fois — egui, l'aperçu GTK, la comparaison GTK — avec les
  mêmes constantes et un ordre d'octets différent (Color32 / RGBA / BGRX).
  `frontend::checker_shade` et `composite_on_checker` fixent la couleur ; chaque
  interface garde son emballage. (`src/frontend.rs`)

### Modifié

- **Champ et curseur fusionnés dans egui** : six réglages (sensibilité variance,
  boost, colonnes, nombre de points, itérations, fréquence de trame)
  occupaient chacun deux lignes — un champ de saisie, puis un curseur sans
  libellé. `egui::Slider` affiche sa valeur *dans* le rail et la garde
  saisissable : une ligne par réglage, et le curseur orphelin disparaît. La
  disposition `[libellé] [rail] [valeur]` est celle des champs de saisie, donc
  le panneau n'affiche plus deux conventions côte à côte.
- **Barre d'état GTK** : la liste des formats d'export est désormais élipsée au
  lieu d'être coupée net dans une fenêtre étroite.
- **« Mémoire estimée : 695 Mo » (GTK)** devient « Pleine résolution : ~695 Mo ».
  Le chiffre seul alarmait sans dire ce qu'il annonce — ce n'est pas la mémoire
  déjà employée, c'est celle que demandera la passe finale.
- **Panneau de réglages redimensionnable (GTK)** : `adw::OverlaySplitView` au
  lieu de `gtk::Paned`, avec `set_propagate_natural_width(false)` sur le
  `ScrolledWindow`. Le `Paned` imposait la largeur naturelle de sa rangée la
  plus large — 540 px — et ne pouvait pas se rétrécir : en fenêtre 820 px,
  l'aperçu ne gardait que 280 px. Un `AdwBreakpoint` à 900 px replie désormais
  le panneau au-dessus de l'aperçu, avec un bouton pour le rouvrir.
  (`src/gtk/controls.rs`, `src/gtk/ui.rs`)
- **Menu principal sous un bouton hamburger (GTK)** : il était accroché au bouton
  « Exporter », alors qu'il ne contenait presque rien d'exportiel — ouvrir un
  dossier, changer d'image, consulter les raccourcis n'avaient aucun autre
  accès. Le bouton Exporter ne garde plus que les exports. (`src/gtk/wire.rs`)
- **« Dossier… » quitte le groupe des presets (GTK)** : il s'agissait d'ouvrir
  un dossier d'images, mais placé entre « Charger un preset » et « Sauver le
  preset », il se lisait comme un dossier de presets. Il est passé dans
  l'en-tête. (`src/gtk/controls.rs`)
- **Réglages conditionnels masqués (GTK)** : « Couleurs de la palette »,
  « Dithering » et « Graine » étaient visibles et insensibles hors de leur
  interrupteur ; egui les masquait déjà. (`src/gtk/sections.rs`)
- **Sous-titres des rangées de curseur retirés (GTK)** : ils affichaient
  « curseur » sous un curseur, et « repère = autre borne » — des notes de
  développement. L'échelle est passée dans l'infobulle.
  (`src/gtk/controls.rs`)
- **Courbes de curseur alignées** : `Itérations` est linéaire partout (le
  logarithmique gaspillait 80 % de la course entre 1 et 10) ; le boost des
  zones uniformes est logarithmique partout (c'est un multiplicateur).
- **Poignée de comparaison contrastée (egui)** : disque blanc et contour gris
  foncé étaient invisibles sur un rendu Voronoï à fond blanc. Halo sombre sous
  un trait clair, comme dans l'interface GTK.
- **Badge « obsolète » sur fond opaque (egui)** : le texte blanc nu était
  illisible sur un ciel clair, et les deux interfaces divergeaient sur ce point.
- **Barre de progression lisible (egui)** : le spinner traversait la barre, dont
  la couleur par défaut disparaissait sur le fond sombre.
- **`Maj+Tab` pour l'aperçu précédent (egui)** : seul `Tab` cyclait, alors que
  la fenêtre de raccourcis GTK annonçait les deux sens. L'action
  « Aperçu précédent » a aussi été ajoutée au menu GTK, et les deux fronts
  sont couverts par un test vérifiant que le cycle boucle dans les deux sens.
- **Liste de formats en français** : « Density map » devient « Carte de
  densité » dans les deux interfaces.
- **Documentation** : `ARCHITECTURE.md` décrit le point de rupture, la palette,
  les notifications, les images récentes et l'ouverture par la ligne de
  commande ; une sous-section explique ce que les deux interfaces ne
  partagent pas, et pourquoi aucune description partagée des réglages n'existe.

- **La poignée basse du rayon de trame pouvait aller jusqu'à 10** : les deux
  interfaces partageaient une seule plage pour les deux poignées du curseur à
  deux bornes, alors que `validate_params` plafonne `halftone_min_radius_ratio`
  à 1,0 quand `halftone_max_dot_ratio` monte à 10. Traîner la poignette au-delà
  de 1 donnait une erreur sèche du worker, sans cause visible. Chaque poignée a
  désormais sa plage (`Bounds` côté egui, borne basse à 1,0 côté GTK) — et le
  test qui manquait est ajouté : l'ancien n'exerçait que la borne basse de la
  borne basse et la borne haute de la borne haute, jamais le sommet de la
  première. (`src/gui/sliders.rs`, `src/gtk/controls.rs`, `src/*/ranges.rs`)
- **Le glisser-déposer GTK ne chargeait que le premier fichier** : le
  `return true` était à l'intérieur de la boucle sur `list.files()`. Déposer un
  dossier et une image ne chargeait que le dossier. (`src/gtk/wire.rs`)
- **`Ctrl+K` et `Espace` mal gardés** : le commentaire annonçait une garde
  `wants_text` que le code ne faisait pas, là où ses huit voisins l'avaient.
  Taper `Ctrl+K` dans un champ de saisie ouvrait la palette en pleine frappe.
  Même trou sur `Espace` côté GTK. (`src/gui/ui.rs`, `src/gtk/wire.rs`)
- **Le GIF divergeait entre la CLI et les interfaces** : `main.rs` tenait sa
  propre liste d'extensions, en avance d'une entrée (`gif`) sur
  `IMAGE_EXTENSIONS`. Un dossier listait donc des fichiers que les interfaces
  refusaient. Les deux viennent de la même constante. (`src/frontend.rs`,
  `src/main.rs`)
- **L'export rapide GTK écrasait sans prévenir** : `quick_export` écrivait
  `{stem}-pointimg.png` à côté de la source sans confirmation, là où egui
  demandait. `src/gtk/confirm.rs` ajoute la demande sur les seuls chemins sans
  dialogue (les autres sont déjà couverts par le portail).
- **Une régression introduite puis retirée au passage** : en coupant les aperçus
  de la CLI, plus aucune itération n'était annoncée — « Iteration n/10 »
  disparaissait. Le contrat interne est maintenant `Option<&RgbImage>`, `None`
  signifiant « pas d'aperçu », et un test verrouille les 7 annonces.
  (`src/filter/mod.rs`)
- **Le test de reproductibilité ne prouvait rien** : il compare des points sur
  un damier, où la densité est nulle partout — les graines ne bougent donc
  jamais, et l'égalité est garantie par construction. (Constat, non corrigé :
  voir « ce qui reste ».)

- **`validate_params` éclaté en sept contrôles nommés**, et les tests qui
  manquaient : chaque contrôle doit nommer le champ qu'il refuse, et la borne
  juste acceptée est vérifiée aussi. L'ordre des vérifications et les messages
  sont inchangés. (`src/filter/params.rs`, `tests/pipeline.rs`)
- **Déduplications restantes** : trois écritures atomiques en une (les copies
  divergeaient déjà sur le type d'erreur et le traitement de l'échec), trois
  blocs « rapport ou échec » en un, deux copies de la paire de rayons GTK en une
  avec sa danse d'emprunts, et l'infobulle des tailles d'export écrite une seule
  fois. (`src/frontend.rs`, `src/gtk/files.rs`, `src/gtk/wire.rs`, `src/export.rs`)

### Corrigé

- **Navigation dans un dossier qui plantait (GTK)** : `load_image` empruntait
  `state` en lecture puis en écriture dans un même `if let` — le temporaire du
  scrutinee vit jusqu'à la fin du bloc, et le second emprunt levait un
  `RefCell already borrowed`. `←`/`→` dans un dossier ouvrait l'image, puis
  interrompait l'application. (`src/gtk/actions.rs`)
- **Recalcul sans changement de valeur** : les deux interfaces ne relancent plus
  un calcul que si les paramètres ont *réellement* changé. GTK réémet
  `value-changed` au retrait du focus d'une `SpinRow`, avec la valeur
  ré-analysée du champ ; egui compare désormais l'ensemble des paramètres avant
  et après le panneau de réglages au lieu de trois champs isolés. Un curseur
  n'est plus ignored… et, à l'inverse, **nombre de points, itérations, colonnes,
  angle de grille, boost, forme des points et palette ne déclenchaient
  aucun calcul** alors que leur valeur changeait. (`src/gui/ui.rs`,
  `src/gtk/wire.rs`)
- **Saisie réécrite à la perte de focus** : egui choisit la précision
  d'affichage d'un `DragValue` selon la distance du pointeur au bord de
  l'écran ; près d'un bord, « Rayon min » s'affichait « 0,000 » puis la saisie
  revenait à 0 (et relançait un calcul). Les décimales affichées sont désormais
  celles de la valeur et de la plage, donc stables. Les entiers s'affichent
  aussi sans décimale parasite (« 800 » et non « 800.0 »).
  (`src/gui/sliders.rs`)
- **Density map recalculée à chaque pixel de glissement** : le calcul de la
  density map part désormais avec le debounce du filtre au lieu d'être lancé à
  chaque variation de la sensibilité. (`src/gui/ui.rs`)
- **Curseur GTK désynchronisé** : un réglage modifié au curseur laissait la
  `SpinRow` jumelle afficher l'ancienne valeur (le curseur GTK n'est pas la
  source de vérité pour `sync_widgets`). (`src/gtk/wire.rs`)
- **Rangées de curseur vides** : masquer un réglage masquait son curseur mais
  laissait sa rangée (vide) visible ; la rangée est désormais mémorisée par le
  curseur. (`src/gtk/sections.rs`)
- **Recalcul sans changement de valeur** : les deux interfaces ne relancent plus
  un calcul que si les paramètres ont *réellement* changé. GTK réémet
  `value-changed` au retrait du focus d'une `SpinRow`, avec la valeur
  ré-analysée du champ ; egui compare désormais l'ensemble des paramètres avant
  et après le panneau de réglages, au lieu de trois champs isolés. À l'inverse,
  **nombre de points, itérations, colonnes, angle de grille, boost, forme des
  points et palette ne déclenchaient aucun calcul** alors que leur valeur
  changeait. (`src/gui/ui.rs`, `src/gtk/wire.rs`)
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
- **Density map calculée pour rien en Quadtree** : `apply_with_progress_inner`
  et `compute_dots_inner` construisaient la density map (deux tables `f64`
  pleine résolution, ~128 Mo sur une image de 8 Mpx) alors que le quadtree
  segmente sur ses propres tables intégrales et ne l'utilise jamais. Le calcul
  est désormais sauté pour cet algorithme. Sortie **bit-identique** (SHA-256
  vérifié avant/après), ~4 % de temps et ~128 Mo de pic mémoire en moins.
  (`src/filter/mod.rs`)

### Dithering : quatre algorithmes au lieu d'un

- `FilterParams::dithering` passe de `bool` à un enum `Dithering` : `Off`,
  `FloydSteinberg`, `Riemersma`, `Bayer4`, `Bayer8`. L'interrupteur des deux
  interfaces devient une liste déroulante, le preset « Calques transparents »
  utilise explicitement Floyd-Steinberg.
- **Riemersma** (nouveau) : diffusion de l'erreur à queue longue le long d'une
  ligne, 16 prises, décroissance 0,7. La file est un décalage pur — chaque part
  d'erreur n'est versée qu'une fois, donc la moyenne est conservée. Calibré sur
  un dégradé de 160 px : l'écart moyen entre pixels voisins passe de 127 (prise
  unique, donc « vers » marqués) à 66, pour un motif un peu plus fondu.
- **Bayer 4×4 et 8×8** (nouveaux) : tramage ordonné, sans diffusion d'erreur.
  Chaque pixel arbitre entre ses deux couleurs de palette les plus proches d'après
  le seuil de la matrice, ce qui donne le rendu « pixel art » / trame. La
  matrice est **générée** par la règle récursive de Bayer (1973)
  (`M(2n) = 4·M(n) + [[0, 2], [3, 1]]`) : aucune table n'est recopiée. Avec une
  palette de deux couleurs, le résultat est un bilevel strict.
- **CLI** : `--dithering [<MODE>]` — sans valeur, l'option vaut Floyd-Steinberg
  (les invocations existantes fonctionnent) ; `--dithering bayer4`, `riemersma`,
  `bayer8` ou `off` choisissent explicitement. (`src/filter/dither.rs`,
  `src/main.rs`)
- **Compatibilité des presets** : les fichiers 0.1 écrivent `dithering = true`,
  qui est toujours lu comme Floyd-Steinberg ; l'écriture passe à
  `dithering = "floyd-steinberg"`. (`src/filter/params.rs`)

### Parité : le rendu GTK n'est plus voilé pendant le calcul

- Le panneau résultat était composé à **55 % d'opacité** et le panneau de
  comparaison à 70 %, dès que le rendu affiché n'était plus final — donc pendant
  tout calcul. Sur fond clair, l'image paraissait visiblement terne pendant
  plusieurs secondes. L'interface egui ne module pas l'opacité, alors que les
  deux calculent `result_is_stale()` à l'identique : l'écart de rendu venait de
  là, et rien ne le documentait. Le voile est retiré ; le caractère provisoire du
  rendu reste signalé par le texte de la barre d'état dans les deux interfaces.
  (`src/gtk/preview.rs`)

### Tests de propriétés sur le dithering, et un piège qu'ils révèlent

- Sept tests de propriétés encadrent les quatre algorithmes : conservation de la
  luminance moyenne sur des images à tonalité variée (six images, cinq largeurs de
  palette, quatre algorithmes) et sur des mélanges de couleurs sans gris dans la
  palette ; proportion de pixels clairs d'une trame ordonnée à la granularité de
  sa matrice ; reproductibilité ; aplat d'une couleur de palette inchangé.
- **Une limite est épinglée volontairement** : une source grise ne peut pas être
  reproduite par une palette devoidée de gris. La sélection du plus proche voisin
  se fait en distance RGB, donc elle ignore la luminance — avec {noir, rouge,
  vert, bleu} l'écart atteint 57 à 73. Le test qui l'atteste échouera le jour où la
  limitation sera corrigée, ce qui est le but.
- **Un piège mesuré en mutation**, consigné dans l'architecture : un test « la
  luminance moyenne est conservée » échoue si l'on inverse le signe des poids,
  mais **passe** si l'on supprime leur normalisation, et **passe** aussi si l'on
  supprime toute propagation d'erreur. La conservation de la moyenne ne prouve
  donc ni la calibration, ni même l'existence d'une diffusion : la diffusion est
  une boucle de rétroaction fermée, et la quantification de chaque pixel
  indépendamment conserve déjà la moyenne d'une rampe monotone. Les contrôles sont
  donc répartis : stabilité par le test de moyenne, lissage (présence et gain)
  par `riemersma_tail_smooths_the_pattern`, calibration par
  `les_poids_de_la_queue_soment_a_un`. Règle retenue : aucune propriété de bout en
  bout n'est considérée comme probante sans avoir été confrontée à une mutation.

- **Riemersma : la file enjambe la fin des lignes.** Elle était remise à zéro après
  chaque ligne, ce qui jetait le résidu en attente et laissait une dérive
  mesurée de 1,0 unité de luminance sur une rampe horizontale (0,83 sur un
  dégradé 2D) ; la dérive est maintenant de 0,00. La file est aussi dimensionnée
  sur `taps` et non sur la largeur de l'image : elle était indexée par colonne,
  donc en fin de ligne la queue se repliait sur des positions déjà traitées, et
  une image plus étroite que la file perdait des parts. Le balayage devient
  **serpentin**, pour que la part d'erreur qui traverse la fin de ligne arrive sur
  le pixel juste en dessous et non au début de la ligne suivante.

### Taille de l'export, mesurée pour de vrai

- La barre d'état affiche désormais `Export : PNG 1.4 Mo · WebP — · JPEG 792 Ko ·
  TIFF 32.0 Mo · BMP 24.0 Mo`, et l'infobulle du bouton d'export rappelle le gain
  par rapport à la source (`source 3.8 Mo · PNG −63 %`). Le bénéfice central de
  `pointimg` — la même image, beaucoup plus légère — était invisible jusqu'au
  premier enregistrement.
- **Les chiffres sont exacts, pas estimés** : l'image est encodée pour de vrai,
  dans le même chemin de code que l'export (`frontend::encode_rgba_image`,
  désormais utilisé aussi bien par la sauvegarde que par la mesure), en parallèle
  avec rayon. Une mesure est faite sur le résultat *définitif* uniquement,
  invalidée dès que l'utilisateur recommence à régler, et publiée hors du thread
  de rendu. (`src/export.rs`, `src/frontend.rs`, `src/gui/exports.rs`,
  `src/gtk/exports.rs`)
- **WebP non mesuré au-delà de 4 Mpx** : l'export WebP est *sans perte*
  (c'est ce que produit `image`), dont l'encodage prend alors plusieurs
  secondes. L'interface affiche « — » plutôt qu'un chiffre inventé.
- `frontend::save_rgba_image` n'écrit plus l'image deux fois (encodage puis
  écriture) : l'encodage se fait une fois en mémoire, puis le remplacement
  atomique du fichier.

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

### Interface graphique — refonte des deux front-ends

Décisions communes à `pointimg-gui` (egui) et `pointimg-gtk` (GTK4/libadwaita),
appliquées après une inspection des deux interfaces en conditions réelles
(fenêtre vide, image chargée, redimensionnement, molette, 1:1).

**Ajouté**

- **Comparaison avant/après par balayage** : nouveau mode par défaut dans les
  deux interfaces ; un curseur sépare la source du résultat dans la même zone
  (double-clic pour le recentrer, glissement pour le déplacer). GTK : rendu
  cairo avec clipping de part et d'autre du curseur ; egui : deux `clip_rect`.
  (`src/gtk/compare.rs`, `src/gui/ui.rs`)
- **Aperçu rapide avec finalisation automatique** : les réglages sont calculés
  sur une image à 50 % ou 25 % (au choix), puis une passe en pleine résolution
  est déclenchée après ~900 ms d'inactivité. Le rendu affiché est signalé
  « obsolète » (opacité réduite + badge) et l'export reste désactivé tant qu'il
  ne l'est pas. Le mode choisi par l'utilisateur n'est plus écrasé par la
  finalisation. (`src/gtk/actions.rs`, `src/gui/compute.rs`)
- **Presets intégrés** : `filter::builtin_presets()` expose neuf points de
  départ validés (pointillisme classique, stippling fin, aplats graphiques,
  rosette CMJN, trame FM, moiré, mosaïque, pastilles, calques transparents),
  proposés dans une liste déroulante par les deux interfaces. Chaque preset est
  testé (`validate_params` + aller-retour TOML). (`src/filter/presets.rs`)
- **Réglages en saisie directe, avec les plages réelles du moteur** :
  `adw::SpinRow` côté GTK (au lieu de curseurs nus) et `DragValue` côté egui.
  Les bornes suivent `validate_params` : 100 000 points, 8 192 colonnes, 100
  itérations, 256 couleurs de palette, 32 canaux halftone, rayon max 1.0 — des
  valeurs auparavant inaccessibles à la souris.
- **Navigation dans un dossier** : « Ouvrir un dossier » puis `←`/`→` pour
  passer d'une image à l'autre ; le titre de la fenêtre affiche la position.
- **Export rapide et presse-papiers** : `Ctrl+E` écrit `<nom>-pointimg.png` à
  côté de la source sans dialogue, `Ctrl+C` copie l'image, et un menu GTK
  regroupe ces actions ainsi que l'export SVG rapide.
- **Bouton « Varier »** : tire une nouvelle graine aléatoire et relance le
  calcul, pour explorer une variante en un clic.
- **Réinitialisation par section** : un bouton dans l'en-tête de chaque groupe
  du panneau GTK rétablit uniquement les réglages de cette section (les autres
  sont conservés) ; l'opération reste annulable.
- **État d'accueil** : au lieu d'un écran vide, les deux interfaces proposent
  « Ouvrir une image », « Ouvrir un dossier » et « Utiliser un exemple » (si
  `assets/examples/` est présent à côté du binaire).
- **Notifications** : les erreurs et les actions (enregistrement, copie) passent
  par des toasts (`adw::Toast`) ou la barre d'état, au lieu de voler la ligne de
  statut qui décrit l'état courant.
- **Titre de fenêtre et barre d'état nommants le fichier** : nom du fichier,
  dimensions et mégapixels, état du calcul, position dans le dossier.
- **Fenêtre de raccourcis** GTK (`Ctrl+?`) et anneau de raccourcis complet
  (`+`/`-`/`0`/`1` zoom, `Tab` vue suivante, `Ctrl+molette`, `F11`,
  `Ctrl+Shift+O`, `Ctrl+E`, `Ctrl+C`, `Ctrl+R`, `Ctrl+P`, `Échap`).
- **Bordure de sécurité autour de la density map** : `State.density_scale`
  mémorise l'échelle du cache ; sans cela, la passe pleine résolution
  réutilisait une density map calculée en ½ et échouait (« density map
  invalide : 2 095 686 valeurs, 8 382 744 attendues »).

**Corrigé**

- **La molette sur le panneau des réglages GTK ne faisait pas défiler le
  panneau** : elle changeait la valeur du réglage situé sous le curseur et
  relançait un calcul à chaque cran (mesuré : « Boost zones uniformes » 2,5 →
  1,0, puis « Sensibilité variance » et « Rayon min » modifiés au passage).
  Un `EventControllerScroll` en phase *capture* redirige maintenant la molette
  vers le `ScrolledWindow` parent ; `Ctrl`/`Shift`/`Alt` ajustent le réglage.
  (`src/gtk/controls.rs`)
- **Les actions importantes étaient sous la ligne de flottaison** : « Annuler »,
  « Export » et la progression étaient dans un panneau de ~1 100 px de haut
  pour 800 px de fenêtre. Elles sont passées dans l'en-tête, avec la progression
  et le bouton d'annulation visibles pendant un long calcul.
- **Le curseur de zoom restait actif en mode « Ajuster »** sans effet et
  affichait une valeur fictive ; il est désormais insensible et le pourcentage
  réellement appliqué est affiché. (`src/gtk/preview.rs`)
- **Passage de « Ajuster » à 1:1** : saut vers le coin haut-gauche ; le centre
  du viewport est désormais conservé.
- **Les vues n'étaient pas synchronisées** : faire défiler « Source » ne
  déplaçait pas « Résultat », rendant la comparaison impossible ; les
  adjustments sont désormais miroités.
- **Double affichage du temps de calcul** dans la barre d'état (« Dernier
  calcul : 671ms » et « Terminé en 671ms. ») ; le message d'état n'est plus
  redondant.
- **L'estimation mémoire était anxiogène et peu parlante** : elle affichait
  « 174 Mo » pour une image de 1,5 Mpx (le plancher de 64 Mio de
  l'estimateur domine). Elle est signalée par un avertissement au-delà de
  1,5 Go, et l'image courante est décrite par ses dimensions.
- **Libellés incohérents** : « Undo »/« Redo » en anglais dans une interface
  française, boutons d'annulation toujours actifs au lieu d'être désactivés,
  « Sensibilité variance » sans indication d'unité.
- **Sections repliées** : « Forme des points » et « Fond et rendu » étaient
  repliées par défaut alors que la forme est le paramètre le plus identitaire
  d'un pointillisme ; toutes les sections sont désormais visibles (chacune tient
  sur une ligne).
- **Redimensionnement d'une `SpinRow` hors de sa plage** : les curseurs min/max
  se désynchronisaient (le rayon max restait à l'ancienne valeur) ; la ligne
  contrainte est réalignée, sans réécrire tous les widgets à chaque cran.
- **Panneau de gauche déborduant sous la barre d'état** côté egui : l'ordre
  d'allocation des panneaux (`haut`, `bas`, puis `gauche`) est corrigé et le
  panneau de réglages est déroulable.
- **Vue comparaison egui affichée à 2 %** : le zoom était déduit d'une hauteur
  calculée une ligne trop tôt ; il est maintenant mesuré sur l'espace réellement
  disponible.
- **Couleurs fausse dans la comparaison egui** : la teinte des deux couches
  était verdâtre (teinte sentinelle au lieu de blanc) ; les images sont
  désormais peintes sans teinte.

**Modifié**

- Le panneau GTK est une `adw::PreferencesPage` : une ligne par réglage
  (`adw::SpinRow`), groupes `adw::PreferencesGroup`, titre et description par
  groupe, `adw::SwitchRow` pour les bascules, `adw::ComboRow` pour les listes.
  Le socle reste libadwaita 1.4 (les widgets 1.6+ — `AdwButtonRow`,
  `AdwMenuButton`, `AdwSpinner` — sont remplacés par leurs équivalents GTK).
- egui : style global appliqué une fois (plus d'air, texte à 13 pt), panneaux
  encadrés, légendes de panneaux, et plus aucun pictogramme Unicode (les
  emojis s'affichaient en « tofu » selon les polices installées).
- egui : `eframe::App::style` n'existe plus en 0.36, le style est donc appliqué
  via `Context::all_styles_mut` au premier rendu ; l'API `Panel::left/top/bottom`
  remplace `SidePanel`/`TopBottomPanel`.

### Curseurs de réglage (les deux interfaces)

- **Saisie + curseur** : les réglages à course large sont doublés d'un curseur
  sous leur champ de saisie. La molette d'un `SpinButton` a un pas fixe (0,1),
  ce qui rend les quatre décimales du rayon min pénibles à atteindre ; le
  curseur se règle à la souris, la saisie chiffrée reste l'outil de la valeur
  exacte. Curseurs ajoutés à la sensibilité, au boost, au nombre de points, aux
  itérations, aux colonnes, à la fréquence de trame, et à la paire min/max des
  rayons (points et halftone).
- **Échelle adaptée à la plage** : logarithmique sur le nombre de points
  (10 → 100 000), les colonnes (2 → 8 192), la fréquence de trame et les rayons
  (0,0005 → 1) ; racine sur la plage du halftone (0 → 10), où le logarithme est
  indéfini et où le linéaire confinerait les valeurs utiles dans le premier
  dixième du rail. La courbe est partagée par les deux interfaces
  (`pointimg::frontend::Curve`) et le curseur arrondit à la précision du champ
  voisin, pour ne pas laisser « 0,25317937 » dans les réglages.
- **Curseur à deux poignées (egui)** : egui n'a pas d'équivalent du curseur à
  deux poignées, le widget est donc dessiné à la main — rail coupé en deux zones
  de saisie disjointes, les bornes ne pouvant pas se croiser. Le segment entre
  les poignées porte la couleur d'accent, la largeur de la plage se lit d'un
  coup d'œil.
- **Paire min/max (GTK)** : GTK n'a pas plus de curseur à deux poignées que
  libadwaita (`AdwSliderRow` n'existe qu'à partir de 1.6). Les deux curseurs
  sont donc empilés, étiquetés « min » et « max », chacun marquant la position de
  l'autre borne ; la contrainte `min ≤ max` reste appliquée, comme sur les
  champs de saisie.
- **Erreur de curseur** : la sensibilité était en échelle logarithmique sur une
  plage commençant à 0 (position indéfinie, poignée collée au maximum) ; elle
  est linéaire.

### Interface graphique (egui) — confort et répartition de l'espace
- **Côte-à-côte rogné** : dans une mise en page horizontale, `available_width()`
  vaut la largeur restante *pour chaque enfant* ; le premier panneau prenait
  toute la place et le second était expulsé hors de la fenêtre (les deux
  images s'affichaient minuscules dans un coin). La place est maintenant
  imposée par l'appelant, et le séparateur est déduit de la largeur disponible
  au lieu d'être compté en double.
- **Empilement automatique** : sous 640 px de largeur centrale, le
  côte-à-côte devient un empilement vertical (deux images de ~300 px de large
  côte à côte sont illisibles).
- **Échelle d'interface** : un réglage « Interface » (85 – 160 %) qui agrandit
  ou réduit d'un coup les textes, les espacements, la hauteur des widgets, la
  largeur des curseurs et les arrondis. Le nouveau style de base est
  confortable (corps 14 pt, interlignes plus généreux, angles arrondis) au
  lieu du rendu par défaut très serré.
- **Thème clair / sombre** commutable depuis le même emplacement (« Auto »
  suit la préférence du système).
- **Préférences d'interface mémorisées** : l'échelle et le thème sont
  enregistrés dès qu'on les change (fichier `ui-prefs.ron` dans le répertoire
  de données de l'application, écrit via l'écriture atomique du projet) et
  relus au lancement. L'`app.ron` d'eframe complète l'ensemble en restaurant
  la taille de la fenêtre et la largeur des panneaux. (`src/gui/prefs.rs`)
- **Barres qui se replient** : l'en-tête d'actions et la rangée
  « Affichage » passent en `horizontal_wrapped`, donc à 160 % ou dans une
  fenêtre étroite les boutons passent à la ligne au lieu d'être coupés.
- La ligne « Interface » est compacte et repliable : elle élargissait le
  panneau de réglages de 415 px et volait de la place à l'aperçu.

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
- **Normalisation de la densité en place** : `normalize_variance` transforme son
  `Vec<f32>` au lieu d'en allouer un second pleine résolution (~33 Mo
  économisés sur une image de 8 Mpx). (`src/filter/density.rs`)
- **API publique entièrement documentée** : `#![warn(missing_docs)]` dans la
  lib (donc bloquant via le `-D warnings` de la CI) ; les 24 items publics
  manquants (`Algorithm`, `FilterParams`, `Dot`, constantes de bornes, API
  colorimétrique, density map…) sont désormais documentés. (`src/lib.rs`,
  `src/filter/params.rs`, `src/color.rs`, `src/filter/density.rs`)
- **Métadonnées du paquet** : `license`, `repository`, `homepage`,
  `documentation`, `readme`, `keywords` et `categories` ajoutés (le warning
  « manifest has no license … » de `cargo package` disparaît) ; `.github` et
  `.idea` exclus du paquet. (`Cargo.toml`)
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

#### Simplifications et ergonomie (egui + GTK4)

- **Sections repliables** : le panneau de réglages n'est plus une liste plate de
  séparateurs. `Ui::collapsing` (egui) et `gtk::Expander` (GTK4) ; « Forme des
  points » et « Fond et rendu » démarrent repliés, le flux courant
  (ouvrir → algorithme → calculer → exporter) tient sans défiler.
- **Infobulles sur tous les paramètres** (≈30 dans chaque frontend) : plus aucun
  réglage n'est une étiquette opaque (« Boost zones uniformes », « Screening
  AM/FM », « Rayon max (frac. step) », description de chaque algorithme…).
- **Plus de contrôles grisés** : le mode halftone affiche désormais son bloc à la
  place des paramètres de placement qu'il ignore, au lieu de les laisser
  visibles mais inactifs. Supprime 4 `add_enabled(!is_halftone, …)` (egui) et le
  `set_sensitive` (GTK4). Le bloc halftone est en outre remonté juste sous le
  sélecteur d'algorithme dans les deux frontends.
- **Barre d'affichage en haut (egui)** : mode d'affichage et zoom quittent le
  panneau de réglages pour une barre au-dessus de l'aperçu — parité avec GTK4,
  et le contrôle est au plus près de son effet.
- **Barre d'état épinglée** : message de statut, mémoire estimée et durée du
  dernier calcul sortent de la zone défilante (`Panel::bottom` en egui, barre
  dédiée sous l'aperçu en GTK4) et restent donc lisibles pendant qu'on parcourt
  les réglages.
- **Algorithme en liste déroulante (egui)** : 5 `selectable_label` empilés
  remplacés par un `ComboBox`, aligné sur le `DropDown` de GTK4.
- **Boutons Undo / Redo dans egui** : la fonction n'était découvrable que par
  `Ctrl+Z` ; les deux boutons existent désormais (parité GTK4).
- **Bouton « Réinitialiser »** (les deux frontends) : rétablit
  `FilterParams::default()` en passant par l'historique, donc annulable.
- **Dialogues d'enregistrement pré-remplis** : dernier dossier utilisé et nom
  dérivé de l'image source (`photo.jpg` → `photo-pointimg.png`).

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
