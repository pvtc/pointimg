# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Preset thumbnails**: the nine built-in presets are now chosen by their
  rendering rather than their name. Each thumbnail is produced with the preset's
  own settings, by `cargo run --example preset_thumbnails`, on a sample image —
  a portrait, the only example in the repository where one can see what a preset
  does to a recognizable subject. egui displays them in a three-column grid, GTK
  in a `GtkFlowBox`; the current thumbnail is framed. Without the
  `assets/presets/` folder, both interfaces fall back to a list of names.
  (`examples/preset_thumbnails.rs`, `src/frontend.rs`, `tests/preset_thumbs.rs`)
- **Recent images in egui**: the home screen offers the most recently opened
  images, kept in the preferences file. GTK reads the same information from
  `GtkRecentManager`. (`src/gui/prefs.rs`, `src/gui/ui.rs`)
- **Collapsible sections in egui**: the settings panel was a column of thirty
  sliders to scroll through entirely to find an option. Four sections
  (`Algorithm`, `Dot shape`, `Background and rendering`, `Computation`) collapse,
  and the collapsed state is remembered across launches.
  (`src/gui/ui.rs`)
- **Command palette** (`Ctrl+K`) in egui: twenty actions with their shortcut
  displayed, unavailable commands greyed out but visible. The interface had
  neither a menu nor a shortcut window — its seventeen shortcuts were only
  discoverable on hover, and the density map as well as fullscreen were
  unreachable. (`src/gui/palette.rs`)
- **Notifications in egui**: errors and confirmations appear above the preview
  (3 s, 8 s for an error). A computation error was invisible during computation
  itself, the status line being replaced by progress.
  (`src/gui/notify.rs`)
- **Recent images in GTK** via `GtkRecentManager`, the desktop list, under the
  home screen buttons. (`src/gtk/recent.rs`)
- **Command-line opening**: `pointimg-gtk photo.jpg` and `pointimg-gui photo.jpg`,
  as well as a folder which activates navigation. GTK goes through `HANDLES_OPEN`,
  so also through file manager activation. (`src/gtk/main.rs`, `src/gui/main.rs`)
- **Desktop entries** (`%F`, so opening from the file manager) in `packaging/`,
  with the `hicolor` icon set. (`packaging/`)
- **Logo**: three primary dots cyan, magenta and yellow of decreasing sizes.
  The SVG is the source, `packaging/icons/regenerate.sh` derives all sizes from
  it; the 256 px PNG is embedded in the egui binary, which has no icon theme.
  (`assets/logo/`)
- **Range tests**: each interface declares its bounds and checks both ends of
  each slider against `validate_params`, plus the value just beyond.
  (`src/gui/ranges.rs`, `src/gtk/ranges.rs`)

- **Preview-free path for the CLI** (`filter::apply_with_iteration_counts`):
  the CLI rendered a full image at every Lloyd iteration, an image its callback
  discarded. It now only announces progress. **1.21× measured** on a
  1000×800 preview / 10 iterations / 4000 points, byte-identical output.
  (`src/filter/mod.rs`, `tests/no_preview_path.rs`)
- **Shared logic extracted** into `frontend`, without touching widgets:
  `random_seed` (the seed drawn by "Vary the seed"), `derived_export_name`
  (five copies), `matching_preset_index` (four copies), `collect_folder_images`
  and `FolderError`, `downsample`, the zoom bounds and step, and a `msg` module
  of thirteen messages. These were written once per interface, each able to
  diverge silently. (`src/frontend.rs`)
- **Transparency checkerboard defined once**: the composite formula was written
  three times — egui, GTK preview, GTK comparison — with the same constants and
  a different byte order (Color32 / RGBA / BGRX). `frontend::checker_shade` and
  `composite_on_checker` fix the color; each interface keeps its wrapping.
  (`src/frontend.rs`)

- **Internationalization of the interfaces**: all user-facing strings (CLI, egui,
  GTK) go through GNU gettext. English catalogue embedded in the binary
  (extracted into a temporary directory at launch), language selector in both
  interfaces, default = system language, persistence. 429 entries in
  `locales/en/LC_MESSAGES/pointimg.po`, compiled with `msgfmt`.
  (`src/frontend.rs`, `locales/`, `src/gui/prefs.rs`, `src/gtk/prefs.rs`)
- **Comments and commit messages in English**: all Rust code comments are
  translated to English, and the 26 narrative commit messages are rewritten in
  the impersonal (no first person, no discovery narrative). The strings displayed
  on screen remain in French.
- **Neutralization of comments**: comments described version history ("an earlier
  version…", "the old `fold/reduce`…") rather than the current state of the
  code; rewritten to state what the code does and the constraint it respects.

### Changed

- **Field and slider merged in egui**: six settings (variance sensitivity, boost,
  columns, number of points, iterations, frame rate) each occupied two lines —
  an input field, then a slider without a label. `egui::Slider` displays its
  value *in* the rail and keeps it editable: one line per setting, and the
  orphan slider disappears. The layout `[label] [rail] [value]` is that of the
  input fields, so the panel no longer shows two conventions side by side.
- **GTK status bar**: the export format list is now ellipsized instead of being
  cut off cleanly in a narrow window.
- **"Estimated memory: 695 MB" (GTK)** becomes "Full resolution: ~695 MB".
  The bare number alarmed without saying what it announces — it is not the
  memory already used, it is what the final pass will require.
- **Resizable settings panel (GTK)**: `adw::OverlaySplitView` instead of
  `gtk::Paned`, with `set_propagate_natural_width(false)` on the
  `ScrolledWindow`. The `Paned` imposed the natural width of its widest row —
  540 px — and could not shrink: in an 820 px window, the preview kept only
  280 px. An `AdwBreakpoint` at 900 px now folds the panel above the preview,
  with a button to reopen it. (`src/gtk/controls.rs`, `src/gtk/ui.rs`)
- **Main menu under a hamburger button (GTK)**: it was attached to the "Export"
  button, although it contained almost nothing export-related — opening a folder,
  changing image, consulting shortcuts had no other access. The Export button now
  keeps only exports. (`src/gtk/wire.rs`)
- **"Folder…" leaves the presets group (GTK)**: it opened an image folder, but
  placed between "Load a preset" and "Save preset" it read as a presets folder.
  It has been moved to the header. (`src/gtk/controls.rs`)
- **Conditional settings hidden (GTK)**: "Palette colors", "Dithering" and
  "Seed" were visible and insensitive outside their toggle; egui already hid
  them. (`src/gtk/sections.rs`)
- **Slider row subtitles removed (GTK)**: they displayed "slider" under a slider,
  and "marker = other bound" — development notes. The scale moved into the
  tooltip. (`src/gtk/controls.rs`)
- **Slider curves aligned**: `Iterations` is linear everywhere (logarithmic
  wasted 80% of the travel between 1 and 10); uniform-zones boost is logarithmic
  everywhere (it is a multiplier).
- **Contrasted comparison handle (egui)**: white disc and dark grey outline were
  invisible on a Voronoi rendering on a white background. Dark halo under a
  light stroke, like in the GTK interface.
- **"obsolete" badge on opaque background (egui)**: bare white text was
  illegible on a light sky, and the two interfaces diverged on this point.
- **Readable progress bar (egui)**: the spinner crossed the bar, whose default
  color disappeared on the dark background.
- **`Shift+Tab` for previous preview (egui)**: only `Tab` cycled, whereas the
  GTK shortcut window announced both directions. The "Previous preview" action
  was also added to the GTK menu, and both directions are covered by a test
  verifying the cycle loops both ways.
- **Localized format list**: "Density map" becomes "Density map" in both
  interfaces.
- **Documentation**: `ARCHITECTURE.md` describes the breakpoint, the palette,
  notifications, recent images and command-line opening; a subsection explains
  what the two interfaces do not share, and why no shared description of the
  settings exists.

- **The low handle of the frame radius could go up to 10**: both interfaces shared
  a single range for the two handles of the two-bound slider, whereas
  `validate_params` caps `halftone_min_radius_ratio` at 1.0 when
  `halftone_max_dot_ratio` rises to 10. Dragging the handle beyond 1 gave a dry
  error from the worker, with no visible cause. Each handle now has its own range
  (`Bounds` on the egui side, lower bound at 1.0 on the GTK side) — and the
  missing test is added: the old one exercised only the lower bound of the lower
  bound and the upper bound of the upper bound, never the top of the first.
  (`src/gui/sliders.rs`, `src/gtk/controls.rs`, `src/*/ranges.rs`)
- **GTK drag-and-drop only loaded the first file**: the `return true` was inside
  the loop over `list.files()`. Dropping a folder and an image loaded only the
  folder. (`src/gtk/wire.rs`)
- **`Ctrl+K` and `Space` poorly guarded**: the comment announced a `wants_text`
  guard that the code did not perform, where its eight neighbors had it. Typing
  `Ctrl+K` in an input field opened the palette mid-typing. Same hole on `Space`
  on the GTK side. (`src/gui/ui.rs`, `src/gtk/wire.rs`)
- **The GIF diverged between the CLI and the interfaces**: `main.rs` kept its own
  extension list, one entry ahead (`gif`) of `IMAGE_EXTENSIONS`. A folder thus
  listed files the interfaces refused. Both now come from the same constant.
  (`src/frontend.rs`, `src/main.rs`)
- **GTK quick export overwrote without warning**: `quick_export` wrote
  `{stem}-pointimg.png` next to the source without confirmation, where egui
  asked. `src/gtk/confirm.rs` adds the prompt on the only paths without a dialog
  (the others are already covered by the portal).
- **A regression introduced then removed along the way**: when cutting CLI
  previews, no iteration was announced anymore — "Iteration n/10" disappeared.
  The internal contract is now `Option<&RgbImage>`, `None` meaning "no preview",
  and a test locks the 7 announcements. (`src/filter/mod.rs`)
- **The reproducibility test proved nothing**: it compares points on a
  checkerboard, where density is zero everywhere — the seeds therefore never
  move, and equality is guaranteed by construction. (Observation, not fixed: see
  "what remains".)

- **`validate_params` split into seven named checks**, and the missing tests:
  each check must name the field it rejects, and the just-accepted bound is
  verified too. The order of checks and the messages are unchanged.
  (`src/filter/params.rs`, `tests/pipeline.rs`)
- **Remaining deduplications**: three atomic writes into one (the copies already
  diverged on error type and failure handling), three "report or fail" blocks
  into one, two copies of the GTK radius pair into one with its borrow dance,
  and the export-size tooltip written only once. (`src/frontend.rs`,
  `src/gtk/files.rs`, `src/gtk/wire.rs`, `src/export.rs`)

### Fixed

- **Folder navigation that crashed (GTK)**: `load_image` borrowed `state` for
  reading then writing in the same `if let` — the scrutinee temporary lives until
  the end of the block, and the second borrow raised a `RefCell already
  borrowed`. `←`/`→` in a folder opened the image, then aborted the application.
  (`src/gtk/actions.rs`)
- **Recomputation without value change**: neither interface relaunches a
  computation unless the parameters have *really* changed. GTK re-emits
  `value-changed` on focus loss of a `SpinRow`, with the re-parsed field value;
  egui now compares the whole set of parameters before and after the settings
  panel instead of three isolated fields. A slider is no longer ignored… and,
  conversely, **number of points, iterations, columns, grid angle, boost, dot
  shape and palette triggered no computation** although their value changed.
  (`src/gui/ui.rs`, `src/gtk/wire.rs`)
- **Input rewritten on focus loss**: egui chooses a `DragValue`'s display
  precision according to the pointer distance to the screen edge; near an edge,
  "Min radius" displayed "0.000" then the input reverted to 0 (and relaunched a
  computation). The displayed decimals are now those of the value and the range,
  hence stable. Integers are also displayed without a stray decimal ("800" not
  "800.0"). (`src/gui/sliders.rs`)
- **Density map recomputed on every glide pixel**: density-map computation now
  leaves with the filter debounce instead of being launched on every
  variance-sensitivity change. (`src/gui/ui.rs`)
- **Desynchronized GTK slider**: a setting changed via the slider left the twin
  `SpinRow` showing the old value (the GTK slider is not the source of truth for
  `sync_widgets`). (`src/gtk/wire.rs`)
- **Empty slider rows**: hiding a setting hid its slider but left its (empty) row
  visible; the row is now remembered by the slider. (`src/gtk/sections.rs`)
- **Recomputation without value change**: neither interface relaunches a
  computation unless the parameters have *really* changed. GTK re-emits
  `value-changed` on focus loss of a `SpinRow`, with the re-parsed field value;
  egui now compares the whole set of parameters before and after the settings
  panel, instead of three isolated fields. Conversely, **number of points,
  iterations, columns, grid angle, boost, dot shape and palette triggered no
  computation** although their value changed. (`src/gui/ui.rs`, `src/gtk/wire.rs`)
  `ScrollArea` of the image area shared the same default salt and received the
  same persisted ID — egui displayed a red error. Each panel now uses its label
  as `id_salt`. (`src/gui/ui.rs`)
- **`--jobs 0` honored in batch**: the flag announced "0 = all CPUs" but the pool
  was built with `num_threads(max(1))`, i.e. a single thread (disguised
  sequential). `num_threads(0)` now lets rayon choose all available CPUs.
  (`src/main.rs`)
- **Clippy `--all-targets`**: `tests/gpu_density.rs` used `(v / 2).min(255)`,
  useless, which failed `-D warnings` as soon as tests were linted.
  (`tests/gpu_density.rs`)
- **`--algorithm halftone` without `--halftone`**: the CLI returned an error
  whereas its comment and documentation announced a default `cmyk`. The `cmyk`
  mode is now activated automatically. (`src/main.rs`)
- **Alpha compositing rounded**: `flatten_to_rgb` truncated (`as u8`) instead of
  rounding, unlike the CLI/GUI path — slight bias toward dark and inconsistency
  between frontends. (`src/filter/util.rs`)
- **GUI SVG export**: the extension is now forced to `.svg` (a file chosen as
  `.png` previously received SVG content). (`src/frontend.rs`)
- **Atomic GTK saves**: the GTK4 frontend now writes via a temporary file +
  atomic replacement, like egui, and rounds alpha compositing. (`src/gtk/files.rs`)
- **Halftone AM**: removal of `height_or_h` (a useless identity) and of the
  inverse-rotation computation immediately discarded; a comment contained Chinese
  characters. (`src/filter/halftone.rs`)
- **Halftone memory estimate**: `estimate_memory_bytes` ignored the coverage maps
  (up to ~1 GB in dominant mode), which strongly underestimated the memory
  announced by the GUI. (`src/filter/util.rs`)
- **ICC buffer over-allocation**: `transform_rgb`/`transform_rgba` reserved
  `source.len() * 3` / `* 4` `f32` elements whereas `source.len()` is already the
  number of components — 3 to 4 times the necessary memory (~500 MB useless for
  a maxed-out RGBA image). (`src/color.rs`)
- **False positive "converted profile"**: with `--input-profile auto` (default)
  and no embedded profile, the CLI/GUI announced a nonexistent colorimetric
  conversion. (`src/color.rs`)
- **Degenerate `importance_sample`**: when `num_points ≥ number of pixels`, all
  excess points collapsed onto the last pixel, degenerating Lloyd/k-means. They
  are now spread over all pixels. (`src/filter/sampling.rs`)
- **Gamma + halftone panics**: three public routes reached `unreachable!()` —
  `apply_dynamic` (gamma + halftone), `apply` (gamma + halftone **dominant**,
  only CMYK was routed) and `apply_with_progress_cached` (halftone without
  gamma). Each now delegates to the dedicated halftone pipeline.
  (`src/filter/mod.rs`)
- **Missing halftone bounds**: `halftone_min_radius_ratio` and
  `halftone_max_dot_ratio` had no ceiling; an arbitrarily large preset or flag
  (`--halftone-max-dot 1e9`) produced dots covering the whole image →
  pathologically slow rendering. Bounded to 1.0 and 10.0. (`src/filter/params.rs`)
- **Useless clone of the density map (GUI)**: the density map was cloned in full
  before being published in the `Arc` — up to ~33 MB per computation. The image
  is now built before the move. (`src/gui/compute.rs`)
- **`density_to_image` rounds** instead of truncating, like the other u8
  conversions in the project. (`src/filter/density.rs`)
- **GTK crash "RefCell already mutably borrowed"**: the `1:1` zoom button kept a
  `state.borrow_mut()` open during `set_value`/`set_active`, whose synchronous
  signals called back `state.borrow()` → panic (abort). The borrow is now
  released before touching the widgets; the texture refreshers were hardened the
  same way. (`src/gtk/wire.rs`, `src/gtk/preview.rs`)
- **Density map computed for nothing in Quadtree**: `apply_with_progress_inner`
  and `compute_dots_inner` built the density map (two full-resolution `f64`
  tables, ~128 MB on an 8 Mpx image) whereas the quadtree segments on its own
  integral tables and never uses it. The computation is now skipped for this
  algorithm. Byte-identical output (SHA-256 verified before/after), ~4% time and
  ~128 MB peak memory saved. (`src/filter/mod.rs`)

### Dithering: four algorithms instead of one

- `FilterParams::dithering` goes from `bool` to an enum `Dithering`: `Off`,
  `FloydSteinberg`, `Riemersma`, `Bayer4`, `Bayer8`. The toggle of both
  interfaces becomes a dropdown, the "Transparent layers" preset explicitly uses
  Floyd-Steinberg.
- **Riemersma** (new): long-tail error diffusion along a line, 16 taps, 0.7
  decay. The queue is a pure shift — each error part is poured only once, so the
  average is preserved. Calibrated on a 160 px gradient: the mean gap between
  neighboring pixels goes from 127 (single tap, hence marked "steps") to 66,
  for a slightly smoother pattern.
- **Bayer 4×4 and 8×8** (new): ordered dithering, no error diffusion. Each pixel
  chooses between its two nearest palette colors according to the matrix
  threshold, giving the "pixel art" / screen look. The matrix is **generated** by
  Bayer's recursive rule (1973) (`M(2n) = 4·M(n) + [[0, 2], [3, 1]]`): no
  table is copied. With a two-color palette, the result is strict bilevel.
- **CLI**: `--dithering [<MODE>]` — without a value, the option defaults to
  Floyd-Steinberg (existing invocations still work); `--dithering bayer4`,
  `riemersma`, `bayer8` or `off` choose explicitly. (`src/filter/dither.rs`,
  `src/main.rs`)
- **Preset compatibility**: 0.1 files write `dithering = true`, which is still
  read as Floyd-Steinberg; writing becomes `dithering = "floyd-steinberg"`.
  (`src/filter/params.rs`)

### Parity: GTK rendering is no longer dimmed during computation

- The result panel was composed at **55% opacity** and the comparison panel at
  70%, as soon as the displayed rendering was no longer final — i.e. during the
  whole computation. On a light background, the image looked visibly dull for
  several seconds. The egui interface does not modulate opacity, although both
  compute `result_is_stale()` identically: the rendering gap came from there, and
  nothing documented it. The veil is removed; the provisional character of the
  rendering is still signalled by the status-bar text in both interfaces.
  (`src/gtk/preview.rs`)

### Property tests on dithering, and a trap they reveal

- Seven property tests frame the four algorithms: conservation of the mean
  luminance on images of varied tone (six images, five palette widths, four
  algorithms) and on color mixes with no grey in the palette; proportion of
  light pixels of an ordered screen at the granularity of its matrix;
  reproducibility; a flat palette color left unchanged.
- **A limit is deliberately pinned**: a grey source cannot be reproduced by a
  palette devoid of grey. Nearest-neighbor selection is done in RGB distance,
  so it ignores luminance — with {black, red, green, blue} the gap reaches 57
  to 73. The test attesting it will fail the day the limitation is fixed, which
  is the point.
- **A trap measured by mutation**, recorded in the architecture: a test "mean
  luminance is preserved" fails if the sign of the weights is flipped, but
  **passes** if their normalization is removed, and **passes** also if all error
  propagation is removed. Preserving the mean therefore proves neither
  calibration nor even the existence of a diffusion: diffusion is a closed
  feedback loop, and quantizing each pixel independently already preserves the
  mean of a monotone ramp. The checks are therefore split: stability by the
  mean test, smoothing (presence and gain) by
  `riemersma_tail_smooths_the_pattern`, calibration by
  `les_poids_de_la_queue_soment_a_un`. Rule retained: no end-to-end property is
  considered probative without having been confronted with a mutation.

- **Riemersma: the queue straddles line ends.** It was reset after each line,
  which discarded the pending residue and left a measured drift of 1.0 luminance
  unit on a horizontal ramp (0.83 on a 2D gradient); the drift is now 0.00. The
  queue is also sized on `taps` rather than image width: it was indexed by
  column, so at end of line the tail folded back onto already-processed
  positions, and an image narrower than the queue lost parts. Scanning becomes
  **serpentine**, so the error part crossing the line end lands on the pixel just
  below instead of the start of the next line.

### Export size, measured for real

- The status bar now displays `Export: PNG 1.4 MB · WebP — · JPEG 792 kB ·
  TIFF 32.0 MB · BMP 24.0 MB`, and the export button's tooltip recalls the gain
  relative to the source (`source 3.8 MB · PNG −63 %`). The central benefit of
  `pointimg` — the same image, much lighter — was invisible until the first
  save.
- **The figures are exact, not estimated**: the image is encoded for real, in the
  same code path as export (`frontend::encode_rgba_image`, now used both by
  saving and by measurement), in parallel with rayon. A measurement is made on
  the *definitive* result only, invalidated as soon as the user starts adjusting
  again, and published outside the rendering thread.
  (`src/export.rs`, `src/frontend.rs`, `src/gui/exports.rs`,
  `src/gtk/exports.rs`)
- **WebP not measured beyond 4 Mpx**: WebP export is *lossless* (that is what
  `image` produces), whose encoding then takes several seconds. The interface
  displays "—" rather than an invented figure.
- `frontend::save_rgba_image` no longer writes the image twice (encoding then
  writing): encoding happens once in memory, then atomic file replacement.

### Added

- **`filter::estimate_memory_bytes_for(w, h, params)`**: estimate accounting for
  halftone coverage maps (4 B/px in CMYK, `n` in dominant). Used by the CLI and
  both frontends. (`src/filter/util.rs`)
- **Expanded image fuzzing**: the `image` target now exercises the full pipeline
  (Grid/Kmeans/Voronoi/Quadtree + SVG export) and dominant FM halftone on an
  image reduced to 64 px. (`fuzz/fuzz_targets/image.rs`)
- **Color management tests**: chunked ICC transformation is compared bit-for-bit
  to a single pass (RGB and RGBA); `importance_sample` distribution test; alpha
  compositing rounding test. (`src/color.rs`, `src/filter/sampling.rs`,
  `src/filter/util.rs`)
- **Halftone anti-panic tests**: `apply_dynamic` gamma+CMYK, `apply`
  gamma+dominant and `apply_with_progress_cached` halftone; halftone radius
  bounds. (`tests/halftone.rs`, `src/filter/mod.rs`)
- **GTK mapping tests**: round-trip `algorithm_index`/`view_index` and
  `shape_index` (pure functions until now untested). (`src/gtk/sections.rs`)

- **`pointimg::frontend` module**: helpers shared between the egui and GTK4
  frontends (formatting, extension normalization, atomic writes, RGBA export,
  presets). Removes duplications `format_duration`/`format_memory`,
  `with_extension`, `flatten_rgba`, `rgb_to_rgba_opaque`. (`src/frontend.rs`)
- **CLI tests**: `parse_bg_color`, `parse_halftone`, `parse_screening`,
  `parse_preview_size`, `resolve_output_path` (substitutions and `--svg`) and
  halftone default. (`src/main.rs`)
- **Palette non-regression tests**: deduplicated quantization is compared
  bit-for-bit to the old reference implementation. (`src/filter/render.rs`)
- **Frontend module tests**: formatting, forced extensions, compositing, preset
  round-trip. (`src/frontend.rs`)

- **GTK4 / libadwaita interface**: new `pointimg-gtk` binary, behind the
  optional `gtk` feature (GTK 4.10+, libadwaita 1.4+). Functional parity with
  the egui GUI (5 algorithms, shapes, palette/dithering, background and
  transparency, gamma, CMYK/dominant AM/FM halftone, density map, zoom/Fit/1:1,
  presets, undo/redo, PNG/JPEG/WebP/BMP/TIFF/SVG export, drag-and-drop,
  `Ctrl+O/S/Z/Y` and `Space` shortcuts, side-by-side preview). Computations run
  in a dedicated thread with progressive preview, debounce and cancellation;
  GDK textures are refreshed on the main thread via an async channel.
  (`src/gtk/*`)
- **GTK4 CI**: dedicated job installing `libgtk-4-dev` and `libadwaita-1-dev`,
  with `clippy -- -D warnings` and release build of the `pointimg-gtk` binary.
- **ICC color profiles**: automatic detection of embedded profiles, input
  conversion to sRGB with `moxcms`, support for `--input-profile`
  (`auto`, `srgb`, `display-p3` or `.icc` file) and for `--output-profile`
  with ICC embedding in PNG/JPEG/WebP/TIFF outputs.
- **Preventive downscale**: images exceeding 8 million pixels or 65 535 pixels
  per side are automatically reduced preserving their ratio, before the costly
  colorimetric transformations. The GUI shows the warning and a working-memory
  estimate; the CLI logs this information.
- **High-precision CMYK halftone**: the `--gamma` path uses `f32` coverages
  after linearization for CMYK halftone, without imposing a costlier `f32`
  pipeline on the classic stippling algorithms.
- **Explicit GUI history**: Undo/Redo now displays the recorded actions
  (algorithm, number of points, palette, background, gamma, etc.).
- **Coupled radius control**: the GUI's Min/Max sliders are grouped and bound
  each other to prevent `min_radius > max_radius`.
- **CI smoke fuzzing**: the image and preset targets are compiled and run with
  100 runs at each CI check; they were also run locally with nightly.
- **Scheduled audit**: a weekly `cargo audit` workflow was added.
- **Versioned presets**: generated presets use `preset_version = 1` and a
  `[params]` table. Old flat TOML presets remain readable and future unknown
  versions are explicitly rejected.
- **GUI worker protection**: computations and density maps are tied to a
  generation. A cancelled or stale worker can no longer publish a result, an
  error or a density map in place of a more recent computation.
- **Performance benchmarks**: Criterion measurements added for CMYK/AM halftone
  and the RGBA path with palette+dithering, throughput expressed in pixels.
- **Strengthened image fuzzing**: the fuzz target uses the `image::Limits` API
  compatible with current versions of the crate.
- **4×4 supersampling anti-aliasing** of dots (`coverage_aa` + `blend_coverage`).
  Dot edges are no longer jagged. (`src/filter/render.rs`)
- **Full validation** of parameters: `cols > 0`, `palette_size >= 2`,
  `variance_sensitivity ∈ [0,1]`, `max_boost >= 1.0`, dimension/pixel bounds.
  (`src/filter/params.rs::validate_params`)
- **Declared MSRV**: Rust 1.95 (`rust-version` in `Cargo.toml`).
- **Modular split** of `src/filter.rs` (2072 lines) into 10 files: `params`,
  `density`, `render`, `svg`, `seedgrid`, `sampling`, `util`,
  `algorithms/{grid,kmeans,voronoi,quadtree}`, `dither`, `gamma`, `halftone`.
- **Integration tests** (`tests/{pipeline,svg,reproducibility,palette,halftone}.rs`)
  and Criterion benches (`benches/filter.rs`).
- **GitHub Actions CI** rewritten: `Swatinem/rust-cache` cache, Ubuntu/macOS/
  Windows matrix, `msrv` job (1.95 + stable), `audit` job, `docs` job
  (`--cfg docsrs`), `build` release job.
- **Transparent background** (`--bg transparent`) with RGBA output via
  `apply_rgba`. The SVG rendering omits the background `<rect>` in transparent
  mode. The GUI preview composites on a checkerboard (Photoshop effect).
- **Gamma correction** (`--gamma`): averages in linear space
  (sRGB→linear→average→sRGB) via a 256-entry LUT.
- **CLI verbosity levels**: `-v` (info), `-vv` (debug), `-q` (warnings only).
  `RUST_LOG` takes priority via `parse_default_env`.
- **TOML presets**: `--preset FILE` / `--save-preset FILE`.
  `FilterParams::to_toml_string` / `from_toml_str` (derives `Serialize`/
  `Deserialize`).
- **Floyd-Steinberg dithering** (`--dithering`) on a quantized palette
  (`src/filter/dither.rs`).
- **Examples folder** `assets/examples/` with `result.png`.
- **README badges**: MSRV 1.95 + examples + docs for the new flags (EN/FR).

### GUI — overhaul of both front-ends

Decisions common to `pointimg-gui` (egui) and `pointimg-gtk` (GTK4/libadwaita),
applied after an inspection of both interfaces in real conditions (empty window,
image loaded, resizing, wheel, 1:1).

**Added**

- **Side-by-side comparison by sweeping**: new default mode in both interfaces;
  a cursor separates the source from the result in the same area (double-click to
  recenter, drag to move). GTK: cairo rendering with clipping on either side of
  the cursor; egui: two `clip_rect`s. (`src/gtk/compare.rs`, `src/gui/ui.rs`)
- **Quick preview with automatic finalization**: settings are computed on an
  image at 50% or 25% (user's choice), then a full-resolution pass is triggered
  after ~900 ms of inactivity. The displayed rendering is marked "obsolete"
  (reduced opacity + badge) and export stays disabled until it is not. The mode
  chosen by the user is no longer overwritten by finalization.
  (`src/gtk/actions.rs`, `src/gui/compute.rs`)
- **Built-in presets**: `filter::builtin_presets()` exposes nine validated
  starting points (classic pointillism, fine stippling, graphic flats, CMYK
  rosette, FM screen, moiré, mosaic, dots, transparent layers), offered in a
  dropdown by both interfaces. Each preset is tested (`validate_params` + TOML
  round-trip). (`src/filter/presets.rs`)
- **Settings entered directly, with the engine's real ranges**:
  `adw::SpinRow` on the GTK side (instead of bare sliders) and `DragValue` on
  the egui side. Bounds follow `validate_params`: 100 000 points, 8 192 columns,
  100 iterations, 256 palette colors, 32 halftone channels, max radius 1.0 —
  values previously unreachable with the mouse.
- **Folder navigation**: "Open a folder" then `←`/`→` to move from one image to
  another; the window title shows the position.
- **Quick export and clipboard**: `Ctrl+E` writes `<name>-pointimg.png` next to
  the source without a dialog, `Ctrl+C` copies the image, and a GTK menu groups
  these actions as well as quick SVG export.
- **"Vary" button**: draws a new random seed and relaunches the computation, to
  explore a variant in one click.
- **Reset by section**: a button in the header of each GTK panel group restores
  only that section's settings (the others are kept); the operation remains
  undoable.
- **Home screen**: instead of an empty screen, both interfaces offer "Open an
  image", "Open a folder" and "Use an example" (if `assets/examples/` is present
  next to the binary).
- **Notifications**: errors and actions (save, copy) go through toasts
  (`adw::Toast`) or the status bar, instead of stealing the status line that
  describes the current state.
- **Window title and status bar naming the file**: file name, dimensions and
  megapixels, computation state, position in the folder.
- **GTK shortcut window** (`Ctrl+?`) and full shortcut ring
  (`+`/`-`/`0`/`1` zoom, `Tab` next view, `Ctrl+wheel`, `F11`,
  `Ctrl+Shift+O`, `Ctrl+E`, `Ctrl+C`, `Ctrl+R`, `Ctrl+P`, `Esc`).
- **Safety border around the density map**: `State.density_scale` remembers the
  cache scale; without it, the full-resolution pass reused a density map
  computed at ½ and failed ("invalid density map: 2 095 686 values, 8 382 744
  expected").

**Fixed**

- **The GTK settings panel wheel did not scroll the panel**: it changed the value
  of the setting under the cursor and relaunched a computation at every notch
  (measured: "Uniform-zone boost" 2.5 → 1.0, then "Variance sensitivity" and
  "Min radius" modified along the way). An `EventControllerScroll` in *capture*
  phase now redirects the wheel to the parent `ScrolledWindow`; `Ctrl`/`Shift`/
  `Alt` adjust the setting. (`src/gtk/controls.rs`)
- **Important actions were below the fold**: "Undo", "Export" and progress were in
  a ~1 100 px panel for an 800 px window. They have been moved to the header,
  with progress and the cancel button visible during a long computation.
- **The zoom slider stayed active in "Fit" mode** without effect and displayed a
  fictitious value; it is now insensitive and the actually applied percentage is
  displayed. (`src/gtk/preview.rs`)
- **Switching from "Fit" to 1:1**: jump to the top-left corner; the viewport
  center is now preserved.
- **Views were not synchronized**: scrolling "Source" did not move "Result",
  making comparison impossible; the adjustments are now mirrored.
- **Double display of computation time** in the status bar ("Last computation:
  671ms" and "Finished in 671ms."); the status message is no longer redundant.
- **Memory estimate was anxiety-inducing and unclear**: it displayed "174 MB" for
  a 1.5 Mpx image (the estimator's 64 MiB floor dominates). It is signalled by
  a warning beyond 1.5 GB, and the current image is described by its dimensions.
- **Inconsistent labels**: "Undo"/"Redo" in English in a French interface,
  cancel buttons always active instead of disabled, "Variance sensitivity"
  without a unit indication.
- **Collapsed sections**: "Dot shape" and "Background and rendering" were
  collapsed by default whereas shape is the most identity-defining parameter of
  a pointillism; all sections are now visible (each fits on one line).
- **Resizing a `SpinRow` outside its range**: the min/max sliders desynchronized
  (max radius stayed at the old value); the constrained row is realigned, without
  rewriting all widgets at every notch.
- **Left panel overflowing under the status bar** on the egui side: the panel
  allocation order (`top`, `bottom`, then `left`) is corrected and the settings
  panel is scrollable.
- **egui comparison view displayed at 2%**: zoom was deduced from a height
  computed one line too early; it is now measured on the space actually
  available.
- **Wrong colors in the egui comparison**: the tint of both layers was greenish
  (sentinel tint instead of white); the images are now painted without tint.

**Changed**

- The GTK panel is an `adw::PreferencesPage`: one row per setting
  (`adw::SpinRow`), `adw::PreferencesGroup` groups, title and description per
  group, `adw::SwitchRow` for toggles, `adw::ComboRow` for lists. The base
  remains libadwaita 1.4 (the 1.6+ widgets — `AdwButtonRow`, `AdwMenuButton`,
  `AdwSpinner` — are replaced by their GTK equivalents).
- egui: global style applied once (more air, 13 pt text), framed panels, panel
  captions, and no more Unicode pictograms (emojis displayed as "tofu" depending
  on installed fonts).
- egui: `eframe::App::style` no longer exists in 0.36, so the style is applied
  via `Context::all_styles_mut` at first render; the `Panel::left/top/bottom` API
  replaces `SidePanel`/`TopBottomPanel`.

### Slider controls (both interfaces)

- **Input + slider**: settings with a wide range are doubled with a slider under
  their input field. A `SpinButton` wheel has a fixed step (0.1), which makes
  the four decimals of the min radius painful to reach; the slider is set with
  the mouse, the numeric input remains the tool for the exact value. Sliders
  added to sensitivity, boost, number of points, iterations, columns, frame rate,
  and the min/max pair of radii (dots and halftone).
- **Scale adapted to the range**: logarithmic on number of points
  (10 → 100 000), columns (2 → 8 192), frame rate and radii
  (0.0005 → 1); square root on the halftone range (0 → 10), where logarithm is
  undefined and where linear would confine useful values to the first tenth of
  the rail. The curve is shared by both interfaces
  (`pointimg::frontend::Curve`) and the slider rounds to the precision of the
  neighboring field, so as not to leave "0.25317937" in the settings.
- **Two-handle slider (egui)**: egui has no equivalent of a two-handle slider,
  so the widget is drawn by hand — rail cut into two disjoint input zones, the
  bounds unable to cross. The segment between the handles carries the accent
  color, the width of the range reads at a glance.
- **Min/max pair (GTK)**: GTK has no more two-handle slider than libadwaita
  (`AdwSliderRow` exists only from 1.6). The two sliders are therefore stacked,
  labelled "min" and "max", each marking the position of the other bound; the
  `min ≤ max` constraint remains applied, as on the input fields.
- **Slider error**: sensitivity was on a logarithmic scale over a range starting
  at 0 (undefined position, handle stuck at maximum); it is now linear.

### GUI (egui) — comfort and space distribution

- **Side-by-side cropped**: in a horizontal layout, `available_width()` equals
  the remaining width *for each child*; the first panel took all the space and
  the second was expelled outside the window (both images displayed tiny in a
  corner). The space is now imposed by the caller, and the separator is deduced
  from the available width instead of being counted twice.
- **Automatic stacking**: below 640 px of central width, side-by-side becomes
  vertical stacking (two ~300 px images side by side are unreadable).
- **Interface scale**: a "UI" setting (85 – 160 %) that enlarges or reduces at
  once texts, spacings, widget heights, slider widths and rounding. The new base
  style is comfortable (14 pt body, more generous line spacing, rounded corners)
  instead of the very tight default rendering.
- **Light / dark theme** switchable from the same place ("Auto" follows the
  system preference).
- **Interface preferences remembered**: scale and theme are saved as soon as they
  change (file `ui-prefs.ron` in the application's data directory, written via
  the project's atomic write) and re-read at launch. eframe's `app.ron`
  completes the set by restoring window size and panel widths.
  (`src/gui/prefs.rs`)
- **Collapsing bars**: the action header and the "Display" row use
  `horizontal_wrapped`, so at 160% or in a narrow window the buttons wrap to the
  next line instead of being cut.
- The "UI" line is compact and collapsible: it widened the settings panel to
  415 px and stole space from the preview.

### Tooling / CI

- **Clippy covers all targets**: CI now lints tests, benches and examples via
  `--all-targets`, which would have detected the lint in
  `tests/gpu_density.rs`.
- **CLI-only coverage**: the clippy and test jobs also run
  `--no-default-features --all-targets`, and the GTK job runs the test suite
  with `--features gtk`. The build without GUI dependencies is thus protected.
  (`.github/workflows/ci.yml`)
- **`rust-toolchain.toml`**: `stable` channel with `rustfmt` and `clippy`
  declared, for a reproducible local environment (MSRV jobs still pin `+1.97`).
- **GTK4 release**: the `pointimg-gtk` binary is built and published
  (`libgtk-4-dev`/`libadwaita-1-dev`) alongside the other artefacts. The egui
  GUI job (`pointimg-gui`) no longer installs `libgtk-3-dev`/`libspeechd-dev`,
  useless: `rfd` 0.17 goes through the XDG portal, and the GTK4 frontend is a
  separate binary.
- **Audit**: the maintenance alert `RUSTSEC-2024-0436` (`paste`, pulled by the
  optional AVIF encoder) is documented in `audit.toml`.
- **Versioned fuzz corpus** (`fuzz/corpus/`) as regression seeds.
- **Reproducible release**: all `release.yml` builds use `--locked`, and
  `softprops/action-gh-release` moves to `v2`. (`.github/workflows/release.yml`)
- **Fuzz repaired in CI**: the root `rust-toolchain.toml` stable overwrote the
  nightly installed by the job, so `cargo fuzz` failed (`-Z` refused on stable).
  A `fuzz/rust-toolchain.toml` (nightly) and `RUSTUP_TOOLCHAIN=nightly` in the
  job guarantee the right toolchain. (`.github/workflows/ci.yml`,
  `fuzz/rust-toolchain.toml`)
- **Real image fuzz seed**: the `image` corpus contained only a 4-byte file
  (never decoded); a small valid PNG (`fuzz/corpus/image/seed-image-12x12`)
  raises the target's coverage from 331 to 10 501 branches.
  (`fuzz/corpus/image/`)
- **Linux release runners moved to Ubuntu 24.04**: the three `release.yml` Linux
  jobs (CLI, egui GUI, GTK4) now run on `ubuntu-24.04` instead of
  `ubuntu-22.04`, so GTK4/libadwaita recent enough for the `v4_14`/`v1_5`
  feature flags can be installed. (`.github/workflows/release.yml`)
- **Native aarch64 builds**: the CLI Linux matrix now uses the
  `ubuntu-24.04-arm` runner instead of cross-compiling from x86_64 with
  `gcc-aarch64-linux-gnu`. (`.github/workflows/release.yml`)
- **MSRV bumped from 1.95 to 1.97**: `rust-version` in `Cargo.toml`, the
  `msrv` CI matrix, the README/ARCHITECTURE badges and the rust-toolchain
  comment are updated. (`.github/workflows/ci.yml`, `Cargo.toml`)
- **GTK feature flags bumped** to `v4_14` (gtk4) and `v1_5` (libadwaita) to
  match the Ubuntu 24.04 system libraries. (`Cargo.toml`)
- **Tests use `assert_matches!`** (stabilised in Rust 1.96) instead of
  `assert!(matches!(...))`. (`src/main.rs`, `src/gtk/actions.rs`)

### Changed

- **CI and docs.rs**: the `gtk` feature is excluded from `--all-features` (GTK4
  and libadwaita are not available on all runners nor on docs.rs). The
  clippy/test/msrv/build/docs jobs now use `--features gui,gpu,avif` and
  `[package.metadata.docs.rs]` explicitly lists these features.
- **Dependency updates**: `egui`/`eframe` 0.31 → 0.36 (trait `App::ui` and
  unified `egui::Panel`), `wgpu` 24 → 30 (instance/adapter/pipeline API),
  `rfd` 0.15 → 0.17, `moxcms` 0.8 → 0.9, `toml` 0.8 → 1, `criterion` 0.5 →
  0.8 (`black_box` via `std::hint`), plus transitive updates. Adaptations in
  `src/gui/{ui,main,convert}.rs`, `src/filter/gpu.rs` and `benches/filter.rs`.
- **K-means acceleration (~15× on bench, byte-identical result)**: nearest-center
  search via `SeedGrid::nearest_by` (lower bound on spatial distance², ties
  broken by index like `min_by`), precomputed normalization LUTs, deterministic
  fixed-chunk f64 accumulation (avoids work-stealing dependence), exact
  fixed-point early stop. (`src/filter/algorithms/kmeans.rs`,
  `src/filter/seedgrid.rs`)
- **Quadtree acceleration** (`src/filter/algorithms/quadtree.rs`): u64 integral
  tables (sums + sums of squares RGB) → O(1) sum/variance queries instead of
  re-walking the node's pixels.
- **Parallel band rendering (rayon)**: `render`/`render_rgba` draw the dots in
  horizontal bands while preserving global order by radius — pixel-identical
  result, several times faster. (`src/filter/render.rs`)
- **`apply()` without wasted previews**: for Kmeans/Voronoi, the non-progressive
  path computes the dots directly then renders once instead of discarding a full
  render at each iteration. (`src/filter/mod.rs`)
- **Memory estimate** updated for the quadtree integral tables
  (6 u64/channel-pixel). (`src/filter/util.rs::estimate_memory_bytes`)
- **Accelerated palette quantization**: dot colors are deduplicated once and the
  nearest center is searched only once per unique color (early exit on exact
  match). The result stays bit-for-bit identical (f64 accumulation in original
  order). (`src/filter/render.rs`)
- **Dominant halftone: memory ÷4**: coverage maps go from `f32` to `u8`
  (1/255 quantization, visually neutral). The peak drops from about 1 GB to
  ~268 MB for `n = 32` on a maxed-out image. (`src/filter/halftone.rs`)
- **Dominant halftone: deduplicated k-means**: pixel colors are deduplicated
  once; the nearest center is searched only once per unique color (instead of
  once per pixel and per iteration), identical result.
  (`src/filter/halftone.rs`)
- **Accelerated SVG export**: direct writing via `write!` into a pre-sized
  `String` (no more `format!` per dot nor `Vec<String>` + `join` for polygons),
  unchanged output. (`src/filter/svg.rs`)
- **Accelerated Floyd-Steinberg dithering**: palette pre-converted to `f32`,
  manual nearest search with early exit on exact match (same results).
  (`src/filter/dither.rs`)
- **In-place density normalization**: `normalize_variance` transforms its
  `Vec<f32>` instead of allocating a second full-resolution one (~33 MB saved
  on an 8 Mpx image). (`src/filter/density.rs`)
- **Fully documented public API**: `#![warn(missing_docs)]` in the lib (hence
  blocking via CI's `-D warnings`); the 24 missing public items (`Algorithm`,
  `FilterParams`, `Dot`, bound constants, colorimetric API, density map…) are
  now documented. (`src/lib.rs`, `src/filter/params.rs`, `src/color.rs`,
  `src/filter/density.rs`)
- **Package metadata**: `license`, `repository`, `homepage`, `documentation`,
  `readme`, `keywords` and `categories` added (the "manifest has no license …"
  warning from `cargo package` disappears); `.github` and `.idea` excluded from
  the package. (`Cargo.toml`)
- **Voronoi refactored**: the weighted Lloyd loop is extracted into `run_lloyd`,
  shared between the progressive and direct paths (previously ~60 duplicated
  lines). (`src/filter/algorithms/voronoi.rs`)
- **Chunked ICC transformation**: intermediate `f32` buffers are limited to
  64 k pixels per call instead of two full-image buffers (24–32 B/px); identical
  output. (`src/color.rs`)
- **Release profile**: `codegen-units = 1` (better cross-module optimization) and
  `strip = true` (symbol-less binaries). (`Cargo.toml`)
- **Completed benchmarks**: halftone `dominant-6-am` and memory estimate
  accounting for halftone. (`benches/filter.rs`)

- **`apply_dynamic` simplified**: flattens then delegates to `apply`, removing
  the gamma-path duplication and the associated panic. (`src/filter/mod.rs`)
- **CLI deduplicated**: `main.rs` reuses `pointimg::frontend` for atomic writes,
  RGBA flattening and alpha detection (removal of ~90 duplicated lines).
  (`src/main.rs`)
- **Consistent `Ctrl+S`**: the dialog now offers WebP/BMP/TIFF like the export
  button, instead of PNG/JPEG only. (`src/gui/ui.rs`)

### Batch processing (`--input` glob/folder, `--output` pattern)

- `--input` expansion: single file, glob (`*`/`?`/`[`), folder (image filter).
- `--output` pattern with substitutions: `{n}` (0-padded index), `{stem}`
  (source name without ext.), `{name}` (full source name).
- Batch continues on a defective file (log error + continue).

### WebP/AVIF outputs

- **WebP** included by default (`.webp` extension).
- **AVIF** opt-in: `cargo build --features avif` (pulls `image/avif` / `ravif`).

### Quick preview (`--preview WxH`)

- Source subsampling (Triangle filter, aspect ratio preserved) before the
  pipeline. 7.7× faster on 2000×2000→300×300.

### Multi-channel halftone (`--halftone`) — CMYK rosette and dominant colors

- **`HalftoneMode`**: `Off`, `Cmyk{angles}` (15°/75°/0°/45° by default),
  `Dominant{n, base_angle_deg}` (N colors extracted via k-means on source
  pixels).
- **`Screening`**: `Am` (grid rotated per channel, radius = coverage) or
  `Fm` (stochastic acceptance + sub-pixel jitter, near-constant radius).
- **Multiply compositor** (`render_halftone`): starts from white paper, applies
  each dot in subtractive mode (Magenta + Yellow = Red).
- **CMYK separation** with partial UCR (removes 50% of the neutral component
  when C+M+Y > 0.3).
- **Dominant angles** spread by `base + i × 180° / n` to avoid moiré.
- CLI flags: `--halftone`, `--screening`, `--halftone-freq`,
  `--halftone-min-radius`, `--halftone-max-dot`.

### Fixed

- **Stale GUI results**: a worker tied to an old image or an old computation can
  no longer publish its result, its dots or its error after a source/parameter
  change.
- **GPU GUI textures**: preview textures are no longer destroyed and recreated
  every frame; they are updated only when a buffer actually changes.
- **Transparent CLI export**: JPEG/BMP are now composited onto the background
  instead of receiving an incompatible RGBA buffer directly.
- **Defensive SVG**: dimensions, parameters and non-finite or invalid
  coordinates/radii are rejected before generation.
- **Density map**: integral-table memory was reduced by using two `f64`
  luminance tables instead of six RGB tables, while keeping stable precision on
  large images.
- **Security audit**: the AccessKit chain responsible for the `quick-xml`
  vulnerabilities had been removed from the `eframe` features; it is re-enabled
  (`accesskit`, screen-reader support) now that `quick-xml` 0.41 fixed the
  advisories. Unnecessary codec dependencies remain disabled by default.
- **Screen angle `--grid-angle`**: grid rotation (Grid algorithm) around the
  image center. Experimental "halftone screen" effect.
- **Uniform validation**: `apply()` now also validates the parameters of the Grid
  and Quadtree algorithms; invalid, non-finite or negative halftone frequencies
  and radii are rejected.
- **Cached density map**: a map of incorrect length returns an error instead of
  causing a panic.
- **RGBA dithering**: the transparent path now applies the same
  Floyd-Steinberg quantization as the RGB path, while preserving alpha.
- **Atomic saves**: temporary files are deleted on error and replacement of
  existing files is handled for Windows.
- **CLI batch**: processing continues after an error, but now returns a failure
  code if at least one file could not be processed.
- **Documented limits**: resource ceilings and parameter scope per algorithm are
  documented in the READMEs and the architecture.

### Changed

- **`Algorithm` derives `clap::ValueEnum`** directly — removal of the `AlgoArg`
  wrapper and of the redundant `From<AlgoArg>` conversions.
- **`Args::to_filter_params()`** extracted: building `FilterParams` from CLI
  flags is centralized (main.rs only routes). The GUI is unchanged (it builds
  `DotShape` directly from sliders).

### Documentation

- **`CHANGELOG.md`** in [Keep a Changelog](https://keepachangelog.com) format.

### GUI

- **Undo / Redo** (`Ctrl+Z` / `Ctrl+Y` or `Ctrl+Shift+Z`): `FilterParams`
  state stack with drag debounce. 50 entries max (FIFO).
- Display of working-memory estimate and warning on automatic image resize.
- Minimum and maximum radii manipulated in a coupled control with dynamic bounds.
- `FilterParams: PartialEq` derived to detect real changes.
- Visual indicator "Cancelled." / "Redone." in the status bar.

#### Simplifications and ergonomics (egui + GTK4)

- **Collapsible sections**: the settings panel is no longer a flat list of
  separators. `Ui::collapsing` (egui) and `gtk::Expander` (GTK4); "Dot shape"
  and "Background and rendering" start collapsed, the current flow
  (open → algorithm → compute → export) fits without scrolling.
- **Tooltips on all parameters** (~30 in each frontend): no setting is an opaque
  label anymore ("Uniform-zone boost", "AM/FM screening", "Max radius (frac.
  step)", description of each algorithm…).
- **No more greyed-out controls**: halftone mode now displays its block in place
  of the placement parameters it ignores, instead of leaving them visible but
  inactive. Removes 4 `add_enabled(!is_halftone, …)` (egui) and the
  `set_sensitive` (GTK4). The halftone block is also moved up just below the
  algorithm selector in both frontends.
- **Display bar on top (egui)**: display mode and zoom leave the settings panel
  for a bar above the preview — parity with GTK4, and the control is as close
  as possible to its effect.
- **Pinned status bar**: status message, estimated memory and last computation
  duration leave the scrolling area (`Panel::bottom` in egui, dedicated bar
  under the preview in GTK4) and therefore remain readable while browsing the
  settings.
- **Algorithm as dropdown (egui)**: 5 stacked `selectable_label`s replaced by a
  `ComboBox`, aligned with GTK4's `DropDown`.
- **Undo/Redo buttons in egui**: the feature was only discoverable via
  `Ctrl+Z`; both buttons now exist (GTK4 parity).
- **"Reset" button** (both frontends): restores `FilterParams::default()` via
  the history, hence undoable.
- **Pre-filled save dialogs**: last used folder and name derived from the source
  image (`photo.jpg` → `photo-pointimg.png`).

### GUI multi-format save

- Output dialog extended: PNG / JPEG / WebP / BMP / TIFF.
- PNG/WebP/TIFF preserve alpha; JPEG/BMP composite onto `bg_color` if
  `--bg transparent`.

### GUI transparent display

- RGBA preview composites on an 8×8 checkerboard (light grey/dark grey,
  Photoshop effect) via `rgba_to_color_image_checker`. The transparent area is
  visible between dots.

### GPU acceleration

- Optional `gpu` feature with WGSL compute shader to compute the local variance
  of the density map in parallel per pixel.
- Explicit activation: `cargo build --features gpu` then `POINTIMG_GPU=1`.
- Automatic CPU/SAT fallback if the runtime option is absent or no GPU
  adapter/device is available.
- GPU tests runnable on an equipped machine:
  `POINTIMG_GPU=1 cargo test --features gpu`.

### GUI halftone/transparent routing

- `start_compute` routes to `apply_rgba` when `transparent` or `halftone != Off`
  (no progressive preview for these two modes — single-pass).

## [0.1.0] — Initial release

- 4 algorithms: Grid, K-means, Voronoi (weighted Lloyd), adaptive Quadtree.
- 4 dot shapes: Circle, Square, Ellipse (aspect + rotation), regular Polygon
  (3-12 sides).
- Density map via summed-area tables (O(W×H)).
- Spatial acceleration `SeedGrid` (2D hash grid + early stopping).
- PNG + vector SVG export.
- Interactive GUI (egui/wgpu) with progressive preview and drag & drop.
- `clap` CLI with all parameters.
- Optional reduced palette (k-means RGB on dots).
- Customizable background: white, black, or `#rrggbb`.
- 26 unit tests + strict clippy.

[Unreleased]: https://github.com/pvtc/pointimg/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pvtc/pointimg/releases/tag/v0.1.0
