//! Interface preferences persisted between two launches for the GTK front-end.
//!
//! GTK has no equivalent of eframe's storage, so the few settings that must
//! survive a restart (the interface language) are kept in a small file under
//! the user configuration directory.

use std::path::PathBuf;

const FILE_NAME: &str = "ui-locale";

/// Path of the preferences file, in the user configuration directory.
fn storage_path() -> Option<PathBuf> {
    let dir = gtk::glib::user_config_dir().join("pointimg");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join(FILE_NAME))
}

/// Loads the memorized language, if any.
pub(crate) fn load_locale() -> Option<pointimg::frontend::Locale> {
    let path = storage_path()?;
    let code = std::fs::read_to_string(path).ok()?;
    Some(pointimg::frontend::Locale::from_code(&code))
}

/// Saves the language immediately.
pub(crate) fn save_locale(locale: pointimg::frontend::Locale) {
    if let Some(path) = storage_path() {
        let _ = std::fs::write(path, locale.code());
    }
}
