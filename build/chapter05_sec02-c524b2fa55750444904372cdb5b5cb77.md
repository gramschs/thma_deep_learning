---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Optimierer, Learning Rate und Batch Size

% TODO: 3–4 einleitende Sätze — konkretes Szenario.
% Vorschlag: Zwei identische Netze, gleiche Daten — eines lernt in 50
% Epochen, das andere divergiert nach drei. Der einzige Unterschied:
% die Learning Rate.

```{admonition} Lernziele
:class: attention
* [ ] Sie können die Optimierer SGD, SGD mit Momentum und Adam unterscheiden und eine begründete Standardwahl treffen.
* [ ] Sie können die Wirkung der Learning Rate experimentell zeigen und eine geeignete Learning Rate systematisch finden.
* [ ] Sie können den Einfluss der Batch Size auf Trainingsverlauf und Rechenzeit erklären.
```

% TODO: Inhaltliche Unterabschnitte, zum Beispiel:
% ## Der wichtigste Hyperparameter: die Learning Rate
% ## SGD, Momentum, Adam
% ## Batch Size
% Brücke Ingenieurwelt: Schrittweitensteuerung kennt man aus
% numerischen Verfahren (z. B. Newton-Verfahren).

## Zusammenfassung und Ausblick

% TODO: Rückblick; Ausblick: Experimente vergleichen erfordert
% Protokollierung (nächste Sektion).
