//! # pointimg
//!
//! Filtre pointilliste : transforme une image en composition de points
//! colorés (stippling Voronoï, k-means, grille, quadtree, ou trame
//! halftone multi-canaux type rosette CMJN).
//!
//! ## Utilisation minimale
//!
//! ```
//! use image::RgbImage;
//! use pointimg::filter::{apply, Algorithm, FilterParams};
//!
//! let src = RgbImage::from_pixel(64, 64, image::Rgb([40, 90, 160]));
//! let params = FilterParams {
//!     algorithm: Algorithm::Grid,
//!     cols: 16,
//!     ..FilterParams::default()
//! };
//! let dst = apply(&src, &params).expect("filtre appliqué");
//! assert_eq!(dst.dimensions(), (64, 64));
//! ```
//!
//! ## Les entrées principales
//!
//! - [`filter::apply`] / [`filter::apply_dynamic`] : image filtrée à partir d'un
//!   `RgbImage` ou d'une `DynamicImage` (RGBA, niveaux de gris, etc.).
//! - [`filter::compute_dots`] : les dots seuls (sans rendu), pour un export
//!   SVG ([`filter::render_svg_from_dots`]) ou un rendu RGBA
//!   ([`filter::render_rgba`]).
//! - [`filter::apply_with_progress`] : variante itérative avec callback de
//!   progression et token d'annulation (utilisée par la GUI).
//! - [`filter::FilterParams`] : tous les réglages, sérialisables en presets
//!   TOML via [`filter::FilterParams::to_toml_string`].
//!
//! ## Reproductibilité
//!
//! Fixer [`filter::FilterParams::rng_seed`] rend le placement des dots
//! déterministe pour un même couple (image, paramètres).

pub mod color;
pub mod filter;
pub mod frontend;
