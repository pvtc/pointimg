//! Interface preferences persisted between two launches.
//!
//! eframe only writes its `Storage` at autosave (every 30 s) or on a clean
//! exit: a preference changed and quit right away would be lost. We therefore
//! keep our own little file — written immediately, through the project's
//! atomic write — in the configuration directory eframe chose for the
//! application.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Name of the preferences file (distinct from eframe's `app.ron`, which the
/// application must not overwrite).
const FILE_NAME: &str = "ui-prefs.ron";
const KEY_SCALE: &str = "ui_scale";
const KEY_THEME: &str = "ui_theme";
const KEY_LOCALE: &str = "ui_locale";

/// Prefix of the recent image entries.
///
/// The file is a simple `key = value` per line, with no list. The entries are
/// therefore numbered — `recent_00`, `recent_01`… — and the zero padding makes
/// them sort in order (the write sorts the lines) even if the list grows past
/// ten items.
const KEY_RECENT: &str = "recent_";

/// Number of recent images kept.
const MAX_RECENT: usize = 10;

/// Key of the `n`-th recent image.
fn recent_key(index: usize) -> String {
    format!("{KEY_RECENT}{index:02}")
}

/// Interface preferences loaded from disk.
pub(crate) struct InterfacePrefs {
    values: HashMap<String, String>,
    path: PathBuf,
}

impl InterfacePrefs {
    /// Loads the preferences; returns `None` if the configuration directory is
    /// inaccessible (the application then runs without memorization).
    pub(crate) fn load() -> Option<Self> {
        let path = storage_path()?;
        Some(Self {
            values: read_pairs(&path),
            path,
        })
    }

    /// Memorized interface scale, if one has been saved.
    pub(crate) fn scale(&self) -> Option<f32> {
        self.values
            .get(KEY_SCALE)?
            .parse::<f32>()
            .ok()
            .filter(|scale| (0.5..=3.0).contains(scale))
    }

    /// Memorized forced theme (`None` = follow the system).
    pub(crate) fn theme(&self) -> Option<egui::Theme> {
        match self.values.get(KEY_THEME).map(String::as_str) {
            Some("dark") => Some(egui::Theme::Dark),
            Some("light") => Some(egui::Theme::Light),
            _ => None,
        }
    }

    /// Memorized interface language (`None` = follow the system).
    pub(crate) fn locale(&self) -> Option<pointimg::frontend::Locale> {
        self.values
            .get(KEY_LOCALE)
            .map(|v| pointimg::frontend::Locale::from_code(v))
    }

    /// Saves immediately: quitting right after a change loses nothing.
    pub(crate) fn save(
        &mut self,
        scale: f32,
        theme: Option<egui::Theme>,
        locale: pointimg::frontend::Locale,
    ) {
        self.values
            .insert(KEY_SCALE.to_string(), format!("{scale:.3}"));
        let theme = match theme {
            Some(egui::Theme::Dark) => "dark",
            Some(egui::Theme::Light) => "light",
            None => "auto",
        };
        self.values.insert(KEY_THEME.to_string(), theme.to_string());
        self.values
            .insert(KEY_LOCALE.to_string(), locale.code().to_string());
        self.flush();
    }

    /// Recently opened images, from the most recent to the oldest.
    ///
    /// Entries whose file has disappeared are discarded: offering an image that
    /// no longer exists is useless.
    pub(crate) fn recent(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = (0..MAX_RECENT)
            .filter_map(|index| self.values.get(&recent_key(index)).cloned())
            .map(PathBuf::from)
            .collect();
        paths.retain(|path| path.is_file());
        paths
    }

    /// Puts an image at the top of the recents and writes the file.
    ///
    /// An image already present moves up without being duplicated — reopening
    /// the same image ten times in a row must not push it ten times.
    pub(crate) fn push_recent(&mut self, path: &Path) {
        let path = path.to_string_lossy().into_owned();
        let mut paths: Vec<String> = (0..MAX_RECENT)
            .map(|index| self.values.get(&recent_key(index)).cloned())
            .collect::<Option<Vec<String>>>()
            .unwrap_or_default();
        paths.retain(|stored| stored != &path);
        paths.insert(0, path);
        paths.truncate(MAX_RECENT);
        // Slots beyond the new length disappear: otherwise, shrinking the
        // list would leave orphan entries behind.
        for index in 0..MAX_RECENT {
            match paths.get(index) {
                Some(value) => {
                    self.values.insert(recent_key(index), value.clone());
                }
                None => {
                    self.values.remove(&recent_key(index));
                }
            }
        }
        self.flush();
    }

    /// Writes the preferences file.
    fn flush(&self) {
        write_pairs(&self.path, &self.values);
    }
}

/// Path of the preferences file, in the configuration directory chosen by
/// eframe for the application.
fn storage_path() -> Option<PathBuf> {
    let dir = eframe::storage_dir("pointimg")?;
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    Some(dir.join(FILE_NAME))
}

/// Reads a `key = value` file, the simplest format that stays readable.
fn read_pairs(path: &Path) -> HashMap<String, String> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    contents
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect()
}

/// Writes the file through a temporary file + atomic replacement, like the
/// rest of the project (`frontend::atomic_text_write`).
fn write_pairs(path: &Path, values: &HashMap<String, String>) {
    let mut contents: Vec<String> = values
        .iter()
        .map(|(key, value)| format!("{key} = {value}"))
        .collect();
    contents.sort();
    let contents = contents.join("\n") + "\n";
    if let Err(error) = pointimg::frontend::atomic_text_write(path, &contents) {
        log::warn!(
            "{}",
            pointimg::frontend::t("Préférences d'interface non enregistrées : {error}")
                .replace("{error}", &error.to_string())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "pointimg-prefs-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ui-prefs.ron");

        assert!(read_pairs(&path).is_empty(), "un fichier absent est vide");

        let mut values = HashMap::new();
        values.insert(KEY_SCALE.to_string(), "1.300".to_string());
        values.insert(KEY_THEME.to_string(), "dark".to_string());
        write_pairs(&path, &values);

        let read = read_pairs(&path);
        assert_eq!(read.get(KEY_SCALE).map(String::as_str), Some("1.300"));
        assert_eq!(read.get(KEY_THEME).map(String::as_str), Some("dark"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scale_out_of_range_is_ignored() {
        let values = HashMap::from([(KEY_SCALE.to_string(), "99".to_string())]);
        let prefs = InterfacePrefs {
            values,
            path: PathBuf::new(),
        };
        assert_eq!(prefs.scale(), None, "une valeur aberrante est ignorée");
    }

    /// A zero-padded numbered list reads back in order once the lines are
    /// sorted — that is what the write guarantees, since it sorts.
    #[test]
    fn les_cles_recentes_se_relisent_dans_lordre() {
        for nombre in [1, 9, 10, 11] {
            let attendu: Vec<String> = (0..MAX_RECENT.min(nombre))
                .map(|index| format!("/img/{index}.png"))
                .collect();
            let mut values = HashMap::new();
            for image in &attendu {
                let index: usize = image
                    .trim_start_matches("/img/")
                    .trim_end_matches(".png")
                    .parse()
                    .unwrap();
                values.insert(recent_key(index), image.clone());
            }
            let prefs = InterfacePrefs {
                values,
                path: PathBuf::new(),
            };

            // What the write produces: the lines sorted by key…
            let ecrit: Vec<String> = {
                let mut lignes: Vec<String> = prefs
                    .values
                    .iter()
                    .filter(|(cle, _)| cle.starts_with(KEY_RECENT))
                    .map(|(cle, valeur)| format!("{cle} = {valeur}"))
                    .collect();
                lignes.sort();
                lignes
            };
            // … and what one reads back from it must be the insertion order.
            let relu: Vec<String> = ecrit
                .iter()
                .map(|ligne| ligne.split_once('=').unwrap().1.trim().to_string())
                .collect();
            assert_eq!(
                relu, attendu,
                "la liste doit se relire dans l'ordre pour {nombre} entrée(s)"
            );
        }
    }

    /// Reopening the same image must not duplicate it.
    #[test]
    fn pousser_deux_fois_ne_duplique_pas() {
        let dir = std::env::temp_dir().join(format!(
            "pointimg-prefs-recent-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let fichier = dir.join("photo.png");
        std::fs::write(&fichier, b"x").unwrap();

        let mut prefs = InterfacePrefs {
            values: HashMap::new(),
            path: dir.join("ui-prefs.ron"),
        };
        prefs.push_recent(&fichier);
        prefs.push_recent(&fichier);
        assert_eq!(
            prefs.recent().len(),
            1,
            "la même image ne doit pas se répéter"
        );

        let autre = dir.join("autre.png");
        std::fs::write(&autre, b"x").unwrap();
        prefs.push_recent(&autre);
        assert_eq!(prefs.recent()[0], autre, "la plus récente passe en tête");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
