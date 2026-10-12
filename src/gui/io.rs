//! File overwrite confirmation (`rfd` dialog, egui-specific).
//!
//! Extension normalization and atomic writes are shared with the GTK4
//! frontend via `pointimg::frontend`.

pub(crate) fn confirm_overwrite(path: &std::path::Path) -> bool {
    if !path.exists() {
        return true;
    }
    matches!(
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Remplacer le fichier ?")
            .set_description(format!("Le fichier '{}' existe déjà.", path.display()))
            .set_buttons(rfd::MessageButtons::YesNo)
            .show(),
        rfd::MessageDialogResult::Yes
    )
}
