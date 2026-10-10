//! Notifications éphémères : erreurs et confirmations.
//!
//! La ligne d'état ne décrit que l'état courant — et pendant un calcul elle
//! affiche la progression, ce qui effaçait tout message. Une erreur de
//! chargement ou d'export était donc invisible exactement au moment où elle
//! comptait. Les notifications passent par-dessus, se superposent quelques
//! secondes, puis s'effacent ; la ligne d'état garde son rôle.
//!
//! L'interface GTK a ses `adw::Toast` (`preview.rs`) : les deux fronts
//! restent indépendants, seule la idée est commune.

use std::time::{Duration, Instant};

use eframe::egui;

/// Durée d'affichage. Une erreur reste plus longtemps qu'une confirmation :
/// elle demande une action, alors qu'une confirmation n'en demande pas.
const INFO_LIFETIME: Duration = Duration::from_secs(3);
const ERROR_LIFETIME: Duration = Duration::from_secs(8);

/// Nature du message : pilote la durée et la couleur.
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

/// Une notification affichée, avec sa date d'expiration.
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

/// Empile les notifications et les dessine dans le coin de l'aperçu.
#[derive(Default)]
pub(crate) struct Toasts {
    items: Vec<Toast>,
    /// Nombre maximal affiché simultanément : au-delà, les plus anciens
    /// disparaissent, sans quoi la pile recouvrirait l'image.
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

    /// Retire les notifications expirées. Renvoie `true` si la pile a changé,
    /// pour ne redessiner que lorsqu'il faut.
    pub(crate) fn expire(&mut self) -> bool {
        let now = Instant::now();
        let before = self.items.len();
        self.items.retain(|toast| toast.expires_at > now);
        before != self.items.len()
    }

    /// Durée avant la prochaine expiration, pour planifier un redessin.
    pub(crate) fn next_expiry(&self) -> Option<Duration> {
        let now = Instant::now();
        self.items
            .iter()
            .map(|toast| toast.expires_at.saturating_duration_since(now))
            .min()
    }

    /// Dessine la pile, ancrée en bas à droite de la place disponible.
    pub(crate) fn show(&self, ui: &mut egui::Ui) {
        if self.items.is_empty() {
            return;
        }
        // Le panneau central entier, et non la place restante : les aperçus ont
        // déjà consommé toute la hauteur disponible.
        let corner = ui.max_rect();
        let spacing = ui.spacing().item_spacing.y;
        // Les hauteurs sont calculées avant le placement : la pile se monte
        // depuis le bas, chaque message au-dessus du précédent.
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
            // Ancrage en bas à droite : `max.x` est le bord droit du message,
            // pas son coin gauche — sinon la pile sortait de l'écran.
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
                // Le rouge d'egui tire sur le orange : on prend la couleur
                // d'avertissement du thème, qui reste lisible sur les deux.
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
