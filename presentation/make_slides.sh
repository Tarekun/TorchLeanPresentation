#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Convert any .webp images to .png for pdflatex compatibility
for webp in imgs/*.webp; do
  [ -f "$webp" ] || continue
  png="${webp%.webp}.png"
  if [ ! -f "$png" ] || [ "$webp" -nt "$png" ]; then
    convert "$webp" "$png"
  fi
done

pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Done: main.pdf"
