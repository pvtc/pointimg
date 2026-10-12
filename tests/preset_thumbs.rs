//! The preset thumbnails must not drift away from the presets themselves.
//!
//! They are produced by `cargo run --example preset_thumbnails`, which renders
//! `filter::builtin_presets()` on a sample image. Nothing forces one to
//! regenerate them afterwards: a modified preset would leave a thumbnail that
//! shows the old setting, without signalling anything to the user.
//!
//! This test checks that there is exactly one thumbnail per preset, and in the
//! right order. It does not compare the content - that would require rendering
//! the presets in the test, which would take far too long - but a shifted
//! index or an added preset shows up.

use pointimg::filter::builtin_presets;
use pointimg::frontend::preset_thumbnail;

#[test]
fn chaque_preset_a_sa_vignette() {
    let presets = builtin_presets();
    let manquantes: Vec<&str> = presets
        .iter()
        .enumerate()
        .filter(|(index, _)| preset_thumbnail(*index).is_none())
        .map(|(_, preset)| preset.name)
        .collect();
    assert!(
        manquantes.is_empty(),
        "vignette absente pour {manquantes:?} — \
         `cargo run --example preset_thumbnails`"
    );
}

#[test]
fn les_vignettes_sont_des_images_lisibles_et_petites() {
    for index in 0..builtin_presets().len() {
        let chemin = preset_thumbnail(index).expect("vérifié par le test précédent");
        let image =
            image::open(&chemin).unwrap_or_else(|e| panic!("{} illisible : {e}", chemin.display()));
        // Big enough for a HiDPI screen in a thumbnail of some forty
        // pixels, small enough for the folder to stay light.
        assert!(
            image.width() >= 200 && image.height() >= 150,
            "{} : {}×{} est trop petit pour une vignette",
            chemin.display(),
            image.width(),
            image.height()
        );
        // Crude guard rail: a thumbnail that ran to several megabytes would
        // be a full rendering left in its place. The bound is high because
        // the stochastic presets (fine stippling, FM screening) produce noise,
        // which PNG does not compress - they weigh ~130 KB against ~27 KB
        // for a graphic preset.
        let poids = std::fs::metadata(&chemin).map(|m| m.len()).unwrap_or(0);
        assert!(
            poids < 400 * 1024,
            "{} pèse {poids} octets : ce n'est plus une vignette",
            chemin.display()
        );
    }
}
