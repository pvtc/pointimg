//! End-to-end tests of the `pointimg` binary.
//!
//! The pipeline is already covered through the library; these tests cover the
//! part that only exists in `main.rs`: argument parsing, input expansion,
//! output-path patterns, batch processing and preset files.

use assert_cmd::Command;
use image::GenericImageView;
use predicates::prelude::*;
use std::path::{Path, PathBuf};

fn write_image(path: &Path, color: [u8; 3]) {
    image::RgbImage::from_pixel(32, 24, image::Rgb(color))
        .save(path)
        .expect("écriture de l'image de test");
}

fn command() -> Command {
    Command::cargo_bin("pointimg").expect("binaire pointimg")
}

#[test]
fn help_lists_the_core_options() {
    command()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--algorithm"))
        .stdout(predicate::str::contains("--num-points"))
        .stdout(predicate::str::contains("--preset"));
}

#[test]
fn renders_one_image_with_the_requested_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    let output = dir.path().join("output.png");
    write_image(&input, [30, 90, 160]);

    command()
        .args([
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--algorithm",
            "grid",
            "--cols",
            "4",
            "--seed",
            "42",
        ])
        .assert()
        .success();

    let rendered = image::open(&output).expect("sortie lisible");
    assert_eq!(rendered.dimensions(), (32, 24));
}

#[test]
fn batch_output_gets_an_index_when_the_pattern_has_no_token() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = dir.path().join("inputs");
    std::fs::create_dir(&inputs).unwrap();
    write_image(&inputs.join("a.png"), [255, 0, 0]);
    write_image(&inputs.join("b.png"), [0, 255, 0]);

    let output = dir.path().join("out.png");
    command()
        .args([
            "--input",
            inputs.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--algorithm",
            "grid",
            "--cols",
            "3",
        ])
        .assert()
        .success();

    let mut produced: Vec<PathBuf> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    produced.sort();
    assert_eq!(produced.len(), 2, "deux fichiers attendus : {produced:?}");
    assert!(produced[0].ends_with("out_0.png"), "{produced:?}");
    assert!(produced[1].ends_with("out_1.png"), "{produced:?}");
}

#[test]
fn batch_parallel_mode_renders_every_file() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = dir.path().join("inputs");
    std::fs::create_dir(&inputs).unwrap();
    for (name, color) in [
        ("a.png", [255, 0, 0]),
        ("b.png", [0, 255, 0]),
        ("c.png", [0, 0, 255]),
    ] {
        write_image(&inputs.join(name), color);
    }

    command()
        .args([
            "--input",
            inputs.to_str().unwrap(),
            "--output",
            dir.path().join("{stem}.png").to_str().unwrap(),
            "--algorithm",
            "voronoi",
            "--num-points",
            "16",
            "--iterations",
            "2",
            "--seed",
            "7",
            "--jobs",
            "2",
        ])
        .assert()
        .success();

    for name in ["a.png", "b.png", "c.png"] {
        assert!(
            dir.path().join(name).is_file(),
            "sortie manquante pour {name}"
        );
    }
}

#[test]
fn preset_is_versioned_and_can_be_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let preset = dir.path().join("preset.toml");
    let input = dir.path().join("input.png");
    let output = dir.path().join("output.png");
    write_image(&input, [10, 20, 30]);

    command()
        .args([
            "--save-preset",
            preset.to_str().unwrap(),
            "--algorithm",
            "grid",
            "--cols",
            "7",
        ])
        .assert()
        .success();

    let contents = std::fs::read_to_string(&preset).unwrap();
    assert!(
        contents.contains("preset_version = 1"),
        "preset versionné attendu : {contents}"
    );

    command()
        .args([
            "--preset",
            preset.to_str().unwrap(),
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(output.is_file());
}

#[test]
fn missing_input_and_invalid_preview_fail_cleanly() {
    command()
        .arg("--output")
        .arg("unused.png")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--input"));

    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.png");
    write_image(&input, [1, 2, 3]);
    command()
        .args([
            "--input",
            input.to_str().unwrap(),
            "--output",
            dir.path().join("out.png").to_str().unwrap(),
            "--preview",
            "not-a-size",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--preview"));
}
