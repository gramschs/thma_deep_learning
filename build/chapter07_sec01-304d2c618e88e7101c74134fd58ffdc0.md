---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Bilder als Tensoren und die Faltung

% TODO: 3–4 einleitende Sätze — konkretes Szenario.
% Vorschlag: Ein Kamerabild aus der Qualitätskontrolle hat 12
% Millionen Pixel. Ein vollvernetztes Netz bräuchte dafür Milliarden
% Gewichte — es muss einen besseren Weg geben.
% Wiederverwendung: Extras skimage aus book_ml4ing (Bilder als
% Arrays) als Vorbereitung verlinken/adaptieren.

```{admonition} Lernziele
:class: attention
* [ ] Sie können den Aufbau eines Bildtensors erklären: Höhe, Breite, Kanäle.
* [ ] Sie können eine Faltungsoperation für ein kleines Beispiel von Hand berechnen.
* [ ] Sie können die Wirkung einfacher Filter (zum Beispiel Kantendetektion) erklären und demonstrieren.
* [ ] Sie können begründen, warum ein vollvernetztes Netz für Bilder unpraktikabel ist.
```

% TODO: Inhaltliche Unterabschnitte, zum Beispiel:
% ## Das Bild als Tensor
% ## Die Faltung
% ## Filter und ihre Wirkung
% TikZ: Filterfenster/rezeptives Feld mit fill=my_yellow hervorheben.

## Zusammenfassung und Ausblick

% TODO: Rückblick; Ausblick: Aus der Faltung wird eine Architektur
% (nächste Sektion).
