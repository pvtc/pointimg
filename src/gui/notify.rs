//! Ephemeral notifications: errors and confirmations.
//!
//! The status line only describes the current state — and during a compute it
//! shows the progress, which erased any message. A loading or export error
//! was therefore invisible exactly when it mattered. The notifications go on
//! top, overlay for a few seconds, then fade away; the status line keeps its
//! role.
//!
//! The GTK interface has its `adw::Toast` (`preview.rs`): the two frontends
//! stay independent, only the idea is shared.

use std::time::{Duration, Instant};

use eframe::egui;

/// Display duration. An error stays longer than a confirmation: it asks for
/// an action, while a confirmation does not.
const INFO_LIFETIME: Duration = Duration::from_secs(3);
const ERROR_LIFETIME: Duration = Duration::from_secs(8);

/// Nature of the message: drives the duration and the color.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToastKind {
    Info,
    Error,
}

impl ToastKind {
    fn lifetime(self) -> Duration {
        match self {
            ToastKind::Info => INFO_LIFETIME,
            ToastKind::Error => ERROR_LIFETIME,
        }
    }
}

/// A displayed notification, with its expiry date.
pub(crate) struct Toast {
    pub text: String,
    pub kind: ToastKind,
    expires_at: Instant,
}

impl Toast {
    fn new(text: impl Into<String>, kind: ToastKind) -> Self {
        Self {
            text: text.into(),
            kind,
            expires_at: Instant::now() + kind.lifetime(),
        }
    }
}

/// Stacks the notifications and draws them in the corner of the preview.
#[derive(Default)]
pub(crate) struct Toasts {
    items: Vec<Toast>,
    /// Maximum number shown at once: beyond it, the oldest ones
    /// disappear, otherwise the stack would cover the image.
    capacity: usize,
}

impl Toasts {
    pub(crate) fn new() -> Self {
        Self {
            items: Vec::new(),
            capacity: 4,
        }
    }

    pub(crate) fn push(&mut self, text: impl Into<String>, kind: ToastKind) {
        self.items.push(Toast::new(text, kind));
        let capacity = self.capacity.max(1);
        if self.items.len() > capacity {
            let excess = self.items.len() - capacity;
            self.items.drain(..excess);
        }
    }

    /// Removes the expired notifications. Returns `true` if the stack changed,
    /// so as to redraw only when needed.
    pub(crate) fn expire(&mut self) -> bool {
        let now = Instant::now();
        let before = self.items.len();
        self.items.retain(|toast| toast.expires_at > now);
        before != self.items.len()
    }

    /// Time before the next expiry, to schedule a redraw.
    pub(crate) fn next_expiry(&self) -> Option<Duration> {
        let now = Instant::now();
        self.items
            .iter()
            .map(|toast| toast.expires_at.saturating_duration_since(now))
            .min()
    }

    /// Draws the stack, anchored to the bottom right of the available space.
    pub(crate) fn show(&self, ui: &mut egui::Ui) {
        if self.items.is_empty() {
            return;
        }
        // The whole central panel, and not the remaining space: the previews
        // have already consumed all the available height.
        let corner = ui.max_rect();
        let spacing = ui.spacing().item_spacing.y;
        // The heights are computed before placement: the stack is built from
        // the bottom, each message above the previous one.
        let mut rects = Vec::with_capacity(self.items.len());
        let mut height_from_bottom = 0.0;
        for toast in &self.items {
            let galley = ui.painter().layout_no_wrap(
                toast.text.clone(),
                ui.style()
                    .text_styles
                    .get(&egui::TextStyle::Body)
                    .cloned()
                    .unwrap_or_else(|| egui::FontId::proportional(14.0)),
                ui.visuals().text_color(),
            );
            let padding = egui::vec2(14.0, 9.0);
            let size = galley.size() + 2.0 * padding + egui::vec2(0.0, 2.0);
            // Bottom right anchoring: `max.x` is the right edge of the message,
            // not its left corner — otherwise the stack ran off the screen.
            let rect = egui::Rect::from_min_size(
                egui::pos2(
                    corner.max.x - size.x,
                    corner.max.y - height_from_bottom - size.y,
                ),
                size,
            );
            height_from_bottom += size.y + spacing;
            rects.push((rect, galley, padding));
        }
        for (toast, (rect, galley, padding)) in self.items.iter().zip(rects) {
            let (fill, text_color) = match toast.kind {
                // egui's red leans toward orange: we take the theme's
                // warning color, which stays readable on both.
                ToastKind::Info => (
                    ui.visuals().panel_fill.gamma_multiply(1.6),
                    ui.visuals().text_color(),
                ),
                ToastKind::Error => (
                    ui.visuals().warn_fg_color.gamma_multiply(0.45),
                    ui.visuals().text_color(),
                ),
            };
            ui.painter().rect_filled(
                rect.expand(2.0),
                egui::CornerRadius::same(6),
                ui.visuals()
                    .noninteractive()
                    .bg_stroke
                    .color
                    .gamma_multiply(0.5),
            );
            ui.painter()
                .rect_filled(rect, egui::CornerRadius::same(5), fill);
            ui.painter()
                .galley(rect.min + padding, galley.clone(), text_color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn la_pile_est_bornee_aux_plus_recents() {
        let mut toasts = Toasts::new();
        for i in 0..10 {
            toasts.push(format!("message {i}"), ToastKind::Info);
        }
        assert_eq!(toasts.items.len(), toasts.capacity);
        assert_eq!(toasts.items.last().unwrap().text, "message 9");
    }

    #[test]
    fn une_erreur_reste_plus_longtemps_quune_confirmation() {
        assert!(ToastKind::Error.lifetime() > ToastKind::Info.lifetime());
    }

    #[test]
    fn une_pile_vide_ne_annonce_aucune_expiration() {
        let mut toasts = Toasts::new();
        assert_eq!(toasts.next_expiry(), None);
        assert!(!toasts.expire());
    }
}
