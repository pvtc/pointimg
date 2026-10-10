#!/bin/sh
# Régénère les icônes depuis assets/logo/pointimg.svg.
#
# Le SVG est la source : les PNG en sont dérivés, pour le thème `hicolor`
# (que le bureau cherche pour un nom d'icône dans le `.desktop`) et pour
# l'icône de fenêtre egui, embarquée dans le binaire.
#
# Usage : sh packaging/icons/regenerate.sh
set -eu

racine=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
svg="$racine/assets/logo/pointimg.svg"
sortie="$racine/packaging/icons/hicolor"

command -v rsvg-convert >/dev/null || {
	echo "rsvg-convert est requis (paquet librsvg2-bin)" >&2
	exit 1
}

mkdir -p "$sortie/scalable/apps"

# Le SVG est la version vectorielle : il suffit de le copier.
cp "$svg" "$sortie/scalable/apps/pointimg.svg"

# Tailles rendues par les lanceurs et les barres de tâches.
for taille in 16 24 32 48 64 128 256 512; do
	dossier="$sortie/${taille}x${taille}/apps"
	mkdir -p "$dossier"
	rsvg-convert -w "$taille" -h "$taille" "$svg" -o "$dossier/pointimg.png"
done

# Copie utilisée par l'interface egui, qui n'a pas de thème d'icônes : le
# fichier est embarqué et lu au démarrage (cf. `src/gui/main.rs`).
cp "$sortie/256x256/apps/pointimg.png" "$racine/assets/logo/pointimg-256.png"

echo "Icônes régénérées dans $sortie"