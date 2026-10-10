//! Les vignettes de presets ne doivent pas dériver des presets eux-mêmes.
//!
//! Elles sont produites par `cargo run --example preset_thumbnails`, qui rend
//! `filter::builtin_presets()` sur une image d'exemple. Rien n'oblige à les
//! régénérer ensuite : un preset modifié laisserait une vignette qui montre
//! l'ancien réglage, sans rien signale à l'utilisateur.
//!
//! Ce test vérifie qu'il y a exactement une vignette par preset, et dans le
//! bon ordre. Il ne compare pas le contenu — cela demanderait de rendre les
//! presets dans le test, ce qui prendrait bien trop de temps — mais un décalage
//! d'index ou un preset ajouté se voit.

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
        // Assez grand pour un écran HiDPI dans une vignette d'une quarantaine
        // de pixels, assez petit pour que le dossier reste léger.
        assert!(
            image.width() >= 200 && image.height() >= 150,
            "{} : {}×{} est trop petit pour une vignette",
            chemin.display(),
            image.width(),
            image.height()
        );
        // Garde-fou grossier : une vignette qui ferait plusieurs mégaoctets
        // serait un rendu complet laissé à la place. La borne est haute parce
        // que les presets stochastiques (stippling fin, trame FM) produisent du
        // bruit, que le PNG ne compresse pas — ils pèsent ~130 Ko contre ~27 Ko
        // pour un preset graphique.
        let poids = std::fs::metadata(&chemin).map(|m| m.len()).unwrap_or(0);
        assert!(
            poids < 400 * 1024,
            "{} pèse {poids} octets : ce n'est plus une vignette",
            chemin.display()
        );
    }
}
