---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Trainingsläufe debuggen

% TODO: 3–4 einleitende Sätze — konkretes Szenario.
% Vorschlag: Die Loss steht nach 100 Epochen exakt dort, wo sie
% angefangen hat. Kein Fehler, keine Warnung — das Netz lernt einfach
% nichts. Wo fängt man an zu suchen?

```{admonition} Lernziele
:class: attention
* [ ] Sie können typische Fehlerbilder beim Training benennen und ihren Ursachen zuordnen: divergierende Loss, konstante Loss, tote Neuronen, fehlerhafte Skalierung.
* [ ] Sie können eine systematische Debugging-Strategie anwenden (zum Beispiel: zuerst auf einen Mini-Datensatz overfitten).
* [ ] Sie können ein fehlerhaftes Trainingssetup eigenständig diagnostizieren und reparieren.
```

% TODO: Inhaltliche Unterabschnitte, zum Beispiel:
% ## Fehlerbilder und ihre Ursachen
% ## Die Overfit-Probe: ein Netz, das nichts lernen kann, ist kaputt
% ## Eine Checkliste für die Fehlersuche
% Übung T1 dieser Woche: absichtlich "kaputtes" Setup reparieren.

## Zusammenfassung und Ausblick

% TODO: Rückblick; Ausblick: Das Training läuft — aber wie gut ist
% das Modell wirklich? (Metriken, nächste Sektion)
