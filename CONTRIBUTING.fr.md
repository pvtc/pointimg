# Contribuer à pointimg

Merci de votre intérêt pour ce projet !

## Signaler un bug

Utilisez les [Issues GitHub](https://github.com/pvtc/pointimg/issues) avec le template "Bug Report".

## Proposer une fonctionnalité

Ouvrez une issue avec le template "Feature Request".

## Pull Requests

1. Fork le repository
2. Créez une branche (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

### Avant de soumettre

- [ ] Le code compile (`cargo build`)
- [ ] Les tests passent (`cargo test --features gui,gpu,avif`)
- [ ] Clippy est content (`cargo clippy --features gui,gpu,avif --all-targets -- -D warnings`)
- [ ] Le formatage est correct (`cargo fmt`)
- [ ] Les fuzz targets compilent (`cargo check --manifest-path fuzz/Cargo.toml`)
- [ ] Les métadonnées livrées valident (`desktop-file-validate`, `appstreamcli`, `flatpak-builder --show-manifest`)

## Développement

```bash
# Build du CLI (par défaut, sans dépendance GUI)
cargo build --release

# Build de la GUI egui
cargo build --release --features gui --bin pointimg-gui

# Tests (le défaut couvre bibliothèque + CLI ; les features ajoutent les GUI)
cargo test
cargo test --features gui,gpu,avif

# Linting
cargo clippy --features gui,gpu,avif --all-targets -- -D warnings

# Formatage
cargo fmt

# Métadonnées livrées
desktop-file-validate packaging/*.desktop
appstreamcli validate --no-net packaging/*.metainfo.xml
flatpak-builder --show-manifest packaging/flatpak/io.github.pvtc.pointimg-gtk.yml > /dev/null

# Fuzzing (installer cargo-fuzz au préalable)
cargo fuzz run preset --fuzz-dir fuzz
cargo fuzz run image --fuzz-dir fuzz
```

Les fuzz targets couvrent le parsing TOML et le décodage d'images non fiables.
Ils appliquent des limites de dimensions et d'allocation pour éviter qu'un corpus
malveillant ne transforme le fuzzing en test de déni de service.

Si vous touchez des chaînes visibles par l'utilisateur, voir le flux i18n
ci-dessous.

## Dépendances

- **MSRV** : Rust 1.100 (`rust-version` dans `Cargo.toml`), vérifié par le job
  CI `msrv` avec `cargo +1.100`.
- **Délai de publication** : `.cargo/config.toml` fait ignorer au résolveur les
  versions publiées depuis moins de 3 jours (`registry.global-min-publish-age`
  et `resolver.incompatible-publish-age = "deny"`). Une version compromise est
  presque toujours retirée dans les heures qui suivent. Les versions déjà dans
  `Cargo.lock` restent valides, mais un `cargo update` global choisira donc des
  versions d'au moins 3 jours ; ciblez une crate avec
  `cargo update -p <crate>` si vous avez besoin d'une version plus récente.
  Pour un correctif urgent, ou une version déjà testée à la main, levez la
  limite le temps d'une commande :

  ```bash
  CARGO_RESOLVER_INCOMPATIBLE_PUBLISH_AGE=allow cargo update -p <crate> --precise <version>
  ```

  Dependabot applique le même délai de 3 jours aux mises à jour de version ;
  les **mises à jour de sécurité ne sont jamais retardées**.
- **Dépendances inutilisées** : `[lints.cargo] unused_dependencies = "deny"`
  fait échouer la compilation quand une dépendance déclarée n'est plus utilisée.

## Traductions (i18n)

Les chaînes visibles par l'utilisateur sont des msgids français passés par
`pointimg::frontend::gettext`. Le catalogue anglais vit dans
`locales/en/LC_MESSAGES/pointimg.po` et se régénère avec :

```bash
sh scripts/update-i18n.sh
```

Le script extrait chaque site d'appel `gettext("...")`, fusionne les nouveaux
msgids dans le `.po`, supprime les obsolètes et recompile le `.mo` embarqué.
Remplissez les nouveaux `msgstr` entre deux exécutions. La CI échoue si le
catalogue commité n'est pas synchronisé avec le code.

## Questions ?

Ouvrez une discussion GitHub ou contactez les mainteneurs.
