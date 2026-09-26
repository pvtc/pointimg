//! Préférences d'interface persistées entre deux lancements.
//!
//! eframe n'écrit son `Storage` qu'à l'autosave (toutes les 30 s) ou à la
//! fermeture propre : une préférence modifiée puis quittée aussitôt serait
//! perdue. On tient donc notre propre petit fichier — écrit immédiatement, via
//! l'écriture atomique du projet — dans le répertoire de configuration que
//! eframe a choisi pour l'application.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Nom du fichier de préférences (distinct de l'`app.ron` d'eframe, que
/// l'application ne doit pas écraser).
const FILE_NAME: &str = "ui-prefs.ron";
const KEY_SCALE: &str = "ui_scale";
const KEY_THEME: &str = "ui_theme";

/// Préférences d'interface chargées depuis le disque.
pub(crate) struct InterfacePrefs {
    values: HashMap<String, String>,
    path: PathBuf,
}

impl InterfacePrefs {
    /// Charge les préférences ; renvoie `None` si le répertoire de
    /// configuration est inaccessible (l'application fonctionne alors sans
    /// mémorisation).
    pub(crate) fn load() -> Option<Self> {
        let path = storage_path()?;
        Some(Self {
            values: read_pairs(&path),
            path,
        })
    }

    /// Échelle d'interface mémorisée, si elle a été enregistrée.
    pub(crate) fn scale(&self) -> Option<f32> {
        self.values
            .get(KEY_SCALE)?
            .parse::<f32>()
            .ok()
            .filter(|scale| (0.5..=3.0).contains(scale))
    }

    /// Thème imposé mémorisé (`None` = suivre le système).
    pub(crate) fn theme(&self) -> Option<egui::Theme> {
        match self.values.get(KEY_THEME).map(String::as_str) {
            Some("dark") => Some(egui::Theme::Dark),
            Some("light") => Some(egui::Theme::Light),
            _ => None,
        }
    }

    /// Enregistre immédiatement : quitter aussitôt après un changement ne perd
    /// rien.
    pub(crate) fn save(&mut self, scale: f32, theme: Option<egui::Theme>) {
        self.values
            .insert(KEY_SCALE.to_string(), format!("{scale:.3}"));
        let theme = match theme {
            Some(egui::Theme::Dark) => "dark",
            Some(egui::Theme::Light) => "light",
            None => "auto",
        };
        self.values.insert(KEY_THEME.to_string(), theme.to_string());
        write_pairs(&self.path, &self.values);
    }
}

/// Chemin du fichier de préférences, dans le répertoire de configuration choisi
/// par eframe pour l'application.
fn storage_path() -> Option<PathBuf> {
    let dir = eframe::storage_dir("pointimg")?;
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    Some(dir.join(FILE_NAME))
}

/// Lit un fichier `clé = valeur`, le format le plus simple qui reste lisible.
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

/// Écrit le fichier par fichier temporaire + remplacement atomique, comme le
/// reste du projet (`frontend::atomic_text_write`).
fn write_pairs(path: &Path, values: &HashMap<String, String>) {
    let mut contents: Vec<String> = values
        .iter()
        .map(|(key, value)| format!("{key} = {value}"))
        .collect();
    contents.sort();
    let contents = contents.join("\n") + "\n";
    if let Err(error) = pointimg::frontend::atomic_text_write(path, &contents) {
        log::warn!("Préférences d'interface non enregistrées : {error}");
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
}
