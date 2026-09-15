//! Écritures atomiques et validation des chemins de sortie.

use std::path::PathBuf;

pub(crate) fn ensure_extension(mut path: PathBuf, default_ext: &str) -> PathBuf {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext)
            if matches!(
                ext.to_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "svg" | "webp" | "bmp" | "tif" | "tiff"
            ) =>
        {
            path
        }
        _ => {
            path.set_extension(default_ext);
            path
        }
    }
}

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

pub(crate) fn temporary_path(path: &std::path::Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output");
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("png");
    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    path.with_file_name(format!(
        ".{}.pointimg-{}-{}-{}.tmp.{}",
        name,
        std::process::id(),
        timestamp,
        id,
        ext
    ))
}

pub(crate) fn atomic_image_save<F>(path: &std::path::Path, save: F) -> image::ImageResult<()>
where
    F: FnOnce(&std::path::Path) -> image::ImageResult<()>,
{
    let tmp = temporary_path(path);
    if let Err(error) = save(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = replace_file(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(image::ImageError::IoError(error));
    }
    Ok(())
}

pub(crate) fn atomic_text_write(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    let tmp = temporary_path(path);
    if let Err(error) = std::fs::write(&tmp, contents) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = replace_file(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    Ok(())
}

pub(crate) fn replace_file(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    #[cfg(windows)]
    if to.exists() {
        std::fs::remove_file(to)?;
    }
    std::fs::rename(from, to)
}
