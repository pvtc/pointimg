//! # pointimg
//!
//! Pointillist filter: turns an image into a composition of colored dots
//! (Voronoi stippling, k-means, grid, quadtree, or a multi-channel halftone
//! screen of the CMYK rosette kind).
//!
//! ## Minimal usage
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
//! ## The main entry points
//!
//! - [`filter::apply`] / [`filter::apply_dynamic`]: a filtered image from an
//!   `RgbImage` or a `DynamicImage` (RGBA, grayscale, etc.).
//! - [`filter::compute_dots`]: the dots alone (without rendering), for an SVG
//!   export ([`filter::render_svg_from_dots`]) or an RGBA render
//!   ([`filter::render_rgba`]).
//! - [`filter::apply_with_progress`]: iterative variant with a progress
//!   callback and a cancellation token (used by the GUI).
//! - [`filter::FilterParams`]: all the settings, serializable as TOML presets
//!   via [`filter::FilterParams::to_toml_string`].
//!
//! ## Reproducibility
//!
//! Setting [`filter::FilterParams::rng_seed`] makes dot placement
//! deterministic for a given (image, parameters) pair.

#![warn(missing_docs)]

pub mod color;
pub mod export;
pub mod filter;
pub mod frontend;
