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
- [ ] Les tests passent (`cargo test`)
- [ ] Clippy est content (`cargo clippy -- -D warnings`)
- [ ] Le formatage est correct (`cargo fmt`)
- [ ] Les fuzz targets compilent (`cargo check --manifest-path fuzz/Cargo.toml`)

## Développement

```bash
# Build
cargo build --release

# Tests
cargo test --lib

# Linting
cargo clippy -- -D warnings

# Formatage
cargo fmt

# Fuzzing (installer cargo-fuzz au préalable)
cargo fuzz run preset --fuzz-dir fuzz
cargo fuzz run image --fuzz-dir fuzz
```

Les fuzz targets couvrent le parsing TOML et le décodage d'images non fiables.
Ils appliquent des limites de dimensions et d'allocation pour éviter qu'un corpus
malveillant ne transforme le fuzzing en test de déni de service.

## Questions ?

Ouvrez une discussion GitHub ou contactez les mainteneurs.
