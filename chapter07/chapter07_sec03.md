---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Transfer Learning

% TODO: 3–4 einleitende Sätze — konkretes Szenario.
% Vorschlag: Für die Qualitätskontrolle haben wir 400 Bilder — viel
% zu wenige, um ein CNN von Grund auf zu trainieren. Ein Netz, das
% bereits auf Millionen Bildern gelernt hat, kann uns aushelfen.

```{admonition} Lernziele
:class: attention
* [ ] Sie können erklären, warum vortrainierte Netze auch mit kleinen Datensätzen gute Ergebnisse liefern.
* [ ] Sie können ein vortrainiertes Netz laden, Schichten einfrieren und auf einen eigenen Datensatz fine-tunen.
* [ ] Sie können ein selbst trainiertes CNN und ein fine-getuntes Netz auf demselben Datensatz vergleichen und das Ergebnis einordnen.
```

```{admonition} Rechenumgebung: GPU empfohlen
:class: warning
Dieses Notebook enthält Trainingsläufe, die auf CPU sehr lange dauern.
Verwenden Sie Google Colab (Laufzeittyp: GPU) oder den JupyterHub der
Hochschule. Eine Anleitung finden Sie in Kapitel 1.
```

% TODO: Inhaltliche Unterabschnitte, zum Beispiel:
% ## Was ein vortrainiertes Netz schon kann
% ## Einfrieren und Fine-Tuning
% ## Der Vergleich: von Grund auf vs. Fine-Tuning
% Wichtigster Hebel für die Projektphase — explizit so benennen.

## Zusammenfassung und Ausblick

% TODO: Rückblick auf Kapitel 7; Ausblick auf Kapitel 8 — von Bildern
% zu Zeitreihen.
