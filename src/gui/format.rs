//! Formatage humain et garde des générations de worker.

use pointimg::filter::FilterParams;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        let secs = ms / 1000;
        format!("{}m {}s", secs / 60, secs % 60)
    }
}

pub(crate) fn format_memory(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} Go", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{} Mo", bytes / (1024 * 1024))
    }
}

pub(crate) fn generation_is_current(generation: &AtomicU64, expected: u64) -> bool {
    generation.load(Ordering::Acquire) == expected
}

pub(crate) fn describe_parameter_change(before: &FilterParams, after: &FilterParams) -> String {
    if before.algorithm != after.algorithm {
        return format!(
            "Algorithme : {:?} → {:?}",
            before.algorithm, after.algorithm
        );
    }
    if before.num_points != after.num_points {
        return format!(
            "Nombre de points : {} → {}",
            before.num_points, after.num_points
        );
    }
    if before.cols != after.cols {
        return format!("Colonnes : {} → {}", before.cols, after.cols);
    }
    if before.iterations != after.iterations {
        return format!("Itérations : {} → {}", before.iterations, after.iterations);
    }
    if before.dot_shape != after.dot_shape {
        return "Forme des points modifiée".to_string();
    }
    if before.palette_size != after.palette_size {
        return "Palette modifiée".to_string();
    }
    if before.bg_color != after.bg_color || before.transparent != after.transparent {
        return "Fond modifié".to_string();
    }
    if before.gamma_correct != after.gamma_correct {
        return "Correction gamma modifiée".to_string();
    }
    if before.halftone != after.halftone || before.screening != after.screening {
        return "Paramètres halftone modifiés".to_string();
    }
    "Paramètres modifiés".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointimg::filter::Algorithm;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn history_labels_changed_algorithm() {
        let before = FilterParams::default();
        let after = FilterParams {
            algorithm: Algorithm::Grid,
            ..before.clone()
        };
        assert!(describe_parameter_change(&before, &after).contains("Algorithme"));
    }

    #[test]
    fn memory_format_is_human_readable() {
        assert_eq!(format_memory(8 * 1024 * 1024), "8 Mo");
    }

    #[test]
    fn stale_worker_generation_is_rejected() {
        let generation = AtomicU64::new(7);
        assert!(generation_is_current(&generation, 7));
        generation.store(8, Ordering::Release);
        assert!(!generation_is_current(&generation, 7));
    }
}
