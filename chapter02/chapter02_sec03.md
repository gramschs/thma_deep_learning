---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Datenaufbereitung: vom Rohdatensatz zum Trainingsdatensatz

% TODO: 3–4 einleitende Sätze — konkretes Szenario.
% Vorschlag: Der Sensordatensatz aus Kapitel 1 enthält Temperaturen in
% Grad Celsius (0–80), Drehzahlen in 1/min (0–6000) und den
% Maschinentyp als Text — so kann kein Modell damit rechnen.

```{admonition} Lernziele
:class: attention
* [ ] Sie können numerische Merkmale skalieren und begründen, warum unterschiedliche Größenordnungen das Training stören.
* [ ] Sie können kategoriale Merkmale mit One-Hot-Encoding in numerische Darstellung überführen.
* [ ] Sie können fehlende Werte in einem Datensatz aufspüren und eine begründete Strategie für den Umgang damit wählen.
* [ ] Sie können einen Aufbereitungs-Workflow so gestalten, dass kein Data Leakage entsteht (Kennwerte nur aus den Trainingsdaten bestimmen).
```

% TODO: Inhaltliche Unterabschnitte, zum Beispiel:
% ## Skalierung und Normalisierung
% ## Kategoriale Merkmale
% ## Fehlende Werte
% ## Der saubere Workflow
% Vertiefung im ML-Buch verlinken (book_ml4ing, Kapitel 5 und 8).

## Zusammenfassung und Ausblick

% TODO: Rückblick auf Kapitel 2; Ausblick auf Kapitel 3 — vom linearen
% Modell zum neuronalen Netz.
