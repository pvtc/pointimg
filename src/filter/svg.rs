use crate::filter::params::{Dot, DotShape, FilterParams};
use crate::filter::render::quantize_dots;
use crate::filter::util::flatten_to_rgb;
use anyhow::{Result, anyhow};
use image::{DynamicImage, RgbImage};
use std::f32::consts::PI;

/// Rend les dots en SVG (chaîne de caractères).
/// Chaque dot devient un élément SVG selon `params.dot_shape`.
///
/// Bug 6 corrigé : utilise `params.bg_color` au lieu de hard-coder blanc/noir.
/// Archi 19 : accepte les dots précalculés pour éviter de relancer le filtre.
pub fn render_svg_from_dots(w: u32, h: u32, dots: &[Dot], params: &FilterParams) -> Result<String> {
    if w == 0 || h == 0 {
        return Err(anyhow!("Image vide"));
    }

    // Q1: avoid cloning dots when no quantization is needed
    let quantized: Vec<Dot>;
    let dots_final: &[Dot] = if let Some(n_colors) = params.palette_size {
        quantized = quantize_dots(dots, n_colors.max(2));
        &quantized
    } else {
        dots
    };

    // Bug 6 corrigé : bg_color comme source de vérité
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#
    );
    svg.push('\n');
    // En mode transparent, on n'émet pas de <rect> de fond : les zones non
    // couvertes par un dot restent transparentes (utile pour compositing SVG).
    if !params.transparent {
        let [br, bg_c, bb] = params.bg_color;
        let bg_hex = format!("#{:02x}{:02x}{:02x}", br, bg_c, bb);
        svg.push_str(&format!(
            r#"  <rect width="{w}" height="{h}" fill="{bg_hex}"/>"#
        ));
        svg.push('\n');
    }

    let mut sorted: Vec<&Dot> = dots_final.iter().collect();
    sorted.sort_unstable_by(|a, b| b.radius.total_cmp(&a.radius));

    for dot in sorted {
        let r = dot.radius;
        if r < 0.5 {
            continue;
        }
        let [cr, cg, cb] = dot.color;
        let fill = format!("#{:02x}{:02x}{:02x}", cr, cg, cb);
        let elem = match params.dot_shape {
            DotShape::Circle => {
                format!(
                    "  <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{fill}\"/>",
                    dot.x, dot.y, r
                )
            }
            DotShape::Square => {
                let s = r * 2.0;
                format!(
                    "  <rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{fill}\"/>",
                    dot.x - r,
                    dot.y - r,
                    s,
                    s
                )
            }
            DotShape::Ellipse { aspect, angle_deg } => {
                let rx = r;
                let ry = (r / aspect.max(0.01)).max(0.5);
                format!(
                    "  <ellipse cx=\"{:.1}\" cy=\"{:.1}\" rx=\"{:.1}\" ry=\"{:.1}\" transform=\"rotate({:.1},{:.1},{:.1})\" fill=\"{fill}\"/>",
                    dot.x, dot.y, rx, ry, angle_deg, dot.x, dot.y
                )
            }
            DotShape::RegularPolygon { sides } => {
                let n = sides.max(3) as usize;
                let pts: String = (0..n)
                    .map(|i| {
                        let a = 2.0 * PI * i as f32 / n as f32 - PI / 2.0;
                        format!("{:.1},{:.1}", dot.x + r * a.cos(), dot.y + r * a.sin())
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("  <polygon points=\"{pts}\" fill=\"{fill}\"/>")
            }
        };
        svg.push_str(&elem);
        svg.push('\n');
    }
    svg.push_str("</svg>\n");
    Ok(svg)
}

/// Rend les dots en SVG depuis une `RgbImage` (recalcule les dots).
/// Pour la sauvegarde CLI — préférer `render_svg_from_dots` en GUI.
pub fn render_svg(src: &RgbImage, params: &FilterParams) -> Result<String> {
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 {
        return Err(anyhow!("Image vide"));
    }
    let dots = crate::filter::compute_dots(src, params)?;
    render_svg_from_dots(w, h, &dots, params)
}

/// Rend les dots en SVG depuis une `DynamicImage` (supporte RGBA).
pub fn render_svg_dynamic(src: &DynamicImage, params: &FilterParams) -> Result<String> {
    let rgb = flatten_to_rgb(src, params.bg_color);
    render_svg(&rgb, params)
}
