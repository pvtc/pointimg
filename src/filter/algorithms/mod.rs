mod grid;
mod kmeans;
mod quadtree;
mod voronoi;

pub(crate) use grid::dots_grid;
pub(crate) use kmeans::{compute_dots_kmeans, dots_kmeans_progressive};
pub(crate) use quadtree::dots_quadtree;
pub(crate) use voronoi::{compute_dots_voronoi, dots_voronoi_progressive};
