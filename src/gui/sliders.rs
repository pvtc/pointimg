//! Adjustment sliders, complements of the input fields.
//!
//! A `DragValue` is driven with the mouse at a fixed sensitivity (0.25 per
//! pixel moved): the same gesture is worth 1.75 iteration on "Iterations" and
//! 0.00175 on "Min radius". The slider gives a travel proportional to the real
//! range, and the logarithmic one a usable precision over wide ranges
//! (number of points: 10 → 100 000). Numeric entry remains the way to
//! indicate an exact value.

use std::ops::RangeInclusive;

use pointimg::frontend::Curve;

/// Decimals needed to write the value without loss.
///
/// The computation is done on the `f32` really stored in `FilterParams`:
/// `0.003f32` is written "0.003", while its conversion to `f64` would give
/// "0.003000011920929" — a display as long as it is misleading.
pub(crate) fn decimals_of(value: f32) -> usize {
    match value.to_string().split_once('.') {
        Some((_, decimals)) => decimals.len(),
        None => 0,
    }
}

/// Display decimals of an adjustment: those of the value, or those of the
/// range if they are finer.
///
/// The result only depends on the range and the value, not on the pointer
/// position: it is essential, since egui picks a display precision on its own
/// depending on the distance from the pointer to the screen edge. A precision
/// coarser than the value would rewrite the entry on loss of focus — and, on
/// such an occasion, trigger a recompute.
pub(crate) fn display_decimals(value: f32, range: &RangeInclusive<f32>) -> usize {
    decimals_of(value)
        .max(decimals_of(*range.start()))
        .max(decimals_of(*range.end()))
}

/// Horizontal slider under an input row: coarse adjustment with the mouse.
/// Rounded to a fixed number of decimals, with no floating point as an
/// intermediate (`(0.1 + 0.2)` is not exactly representable).
pub(crate) fn round_to(value: f32, decimals: usize) -> f32 {
    let factor = 10f64.powi(decimals as i32);
    ((f64::from(value) * factor).round() / factor) as f32
}

/// Range of each handle of the two-ended slider.
///
/// Two ranges rather than one, because the two ends do not share the same
/// ceiling: `validate_params` caps `halftone_min_radius_ratio` at 1.0 when
/// `halftone_max_dot_ratio` goes up to 10. Sharing them let the low handle
/// cross the ceiling — a dry error from the worker, with no visible cause.
///
/// `RangeInclusive` is not `Copy`, so `Bounds` is not either; it does however
/// go by value, which copies two ranges and nothing more.
#[derive(Clone)]
pub(crate) struct Bounds {
    pub low: RangeInclusive<f32>,
    pub high: RangeInclusive<f32>,
}

impl Bounds {
    /// The two handles share the range — the common case.
    pub(crate) fn both(range: RangeInclusive<f32>) -> Self {
        Self {
            low: range.clone(),
            high: range,
        }
    }
}

/// Two-handle slider for a pair of ends (Min radius / Max radius).
///
/// egui has no equivalent of the two-handle `GtkRange`: the widget is drawn by
/// hand. The rail is split into two disjoint input zones, separated halfway
/// between the handles: each handle is therefore driven independently, with no
/// state to keep from one frame to the next, and the two cannot cross each
/// other (the low end never goes past the high one).
///
/// The rail covers the union of both ranges, so that the two handles stay
/// comparable; each one is then brought back into *its* range.
///
/// Returns `true` if the pair changed — numeric entry in the two neighboring
/// rows stays the way to indicate an exact value.
pub(crate) fn range_slider(
    ui: &mut egui::Ui,
    id: &str,
    low: &mut f32,
    high: &mut f32,
    bounds: Bounds,
    curve: Curve,
    hint: &str,
) -> bool {
    let (low_lo, low_hi) = (*bounds.low.start(), *bounds.low.end());
    let (high_lo, high_hi) = (*bounds.high.start(), *bounds.high.end());
    let (start, end) = (low_lo.min(high_lo), low_hi.max(high_hi));
    let to_pos =
        |value: f32| curve.position(f64::from(value), f64::from(start), f64::from(end)) as f32;
    let to_value = |t: f32| curve.value(f64::from(t), f64::from(start), f64::from(end)) as f32;

    let id = egui::Id::new(id);
    let height = 18.0;
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    if !ui.is_rect_visible(rect) || rect.width() <= 1.0 {
        return false;
    }

    let x_of = |t: f32| egui::lerp(rect.left()..=rect.right(), t);
    let (x_low, x_high) = (x_of(to_pos(*low)), x_of(to_pos(*high)));
    let accent = ui.visuals().selection.bg_fill;
    let radius = egui::CornerRadius::same(3);
    let painter = ui.painter_at(rect);
    let track = egui::Rect::from_center_size(rect.center(), egui::vec2(rect.width(), 4.0));
    // Same colors as egui's simple slider: gray rail, light handle.
    let widgets = &ui.visuals().widgets;
    let (track_color, handle_color) = (widgets.inactive.bg_fill, widgets.active.bg_fill);
    painter.rect_filled(track, radius, track_color);
    // The segment between the two handles carries the accent color: the
    // width of the range can be read at a glance.
    painter.rect_filled(
        egui::Rect::from_x_y_ranges(x_low..=x_high, track.y_range()),
        radius,
        accent,
    );
    for x in [x_low, x_high] {
        let center = egui::pos2(x, rect.center().y);
        // The outer disk isolates the handle from the rail, the inner one marks it.
        painter.circle_filled(center, 6.5, track_color);
        painter.circle_filled(center, 4.5, handle_color);
    }

    // Two disjoint zones, separated halfway between the handles: the left zone
    // drives the low end, the right one the high end.
    let middle = (x_low + x_high) / 2.0;
    let low_response = ui.interact(
        egui::Rect::from_x_y_ranges(rect.left()..=middle, rect.y_range()),
        id.with("low"),
        egui::Sense::click_and_drag(),
    );
    let high_response = ui.interact(
        egui::Rect::from_x_y_ranges(middle..=rect.right(), rect.y_range()),
        id.with("high"),
        egui::Sense::click_and_drag(),
    );

    let before = (*low, *high);
    // Same rounding as the simple sliders: a dragged handle must not leave a
    // seven-decimal value in the neighboring field. We keep the coarsest
    // precision of the two ends, so that each one's input zone stays stable.
    let decimals = display_decimals(*low, &bounds.low).max(display_decimals(*high, &bounds.high));
    for (response, is_low) in [(low_response.clone(), true), (high_response.clone(), false)] {
        if !response.dragged() {
            continue;
        }
        let Some(pointer) = response.interact_pointer_pos() else {
            continue;
        };
        let t = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        // The raw value of the rail is then brought back into the range of
        // *this* handle: otherwise the low handle, which can share the high end
        // of the high handle, would cross the ceiling of `validate_params`.
        let raw = round_to(to_value(t), decimals);
        if is_low {
            let borne = raw.clamp(low_lo, low_hi);
            *low = borne.min(*high);
        } else {
            let borne = raw.clamp(high_lo, high_hi);
            *high = borne.max(*low);
        }
    }
    if low_response.hovered() || high_response.hovered() {
        low_response.on_hover_text(hint);
        high_response.on_hover_text(hint);
    }

    (*low, *high) != before
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_of_counts_written_decimals() {
        // Written in `f32`: 0.1 has no stray decimal as in `f64`.
        assert_eq!(decimals_of(800.0), 0);
        assert_eq!(decimals_of(0.1), 1);
        assert_eq!(decimals_of(0.003), 3);
        assert_eq!(decimals_of(0.0005), 4);
        assert_eq!(decimals_of(-180.0), 0);
    }

    #[test]
    fn display_decimals_follows_the_finest_of_value_and_range() {
        // 0.06 in a range that goes down to 0.0005: it is the finest end that
        // imposes the display, otherwise the entry would be rewritten.
        assert_eq!(display_decimals(0.06, &(0.0005..=1.0)), 4);
        // An integer on an integer range displays without a decimal.
        assert_eq!(display_decimals(800.0, &(10.0..=100_000.0)), 0);
        // A value finer than the range keeps its precision.
        assert_eq!(display_decimals(0.12345, &(0.0..=1.0)), 5);
    }

    #[test]
    fn round_to_keeps_the_requested_decimals() {
        assert_eq!(round_to(0.25317937, 4), 0.2532);
        assert_eq!(round_to(0.25317937, 2), 0.25);
        assert_eq!(round_to(799.6, 0), 800.0);
    }
}
