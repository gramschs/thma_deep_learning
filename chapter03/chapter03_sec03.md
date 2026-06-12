---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Gradientenabstieg und Backpropagation

% TODO: 3–4 einleitende Sätze — konkretes Szenario.
% Vorschlag: Unser Netz hat 10.000 Gewichte. Wir drehen an einem und
% die Loss ändert sich — aber an welchem sollen wir drehen, und in
% welche Richtung?
% Dreischritt: konkretes Drehen an Gewichten → Empfindlichkeit der
% Loss → Gradient/Kettenregel. Brücke Ingenieurwelt:
% Sensitivitätsanalyse.
% Hinweis Format: hoher Theorieanteil, Tafel/Folie statt Live-Coding.

```{admonition} Lernziele
:class: attention
* [ ] Sie können die Idee des Gradientenabstiegs anhand einer Loss-Landschaft erklären.
* [ ] Sie können die Rolle der Learning Rate erklären und die Folgen einer zu großen oder zu kleinen Learning Rate beschreiben.
* [ ] Sie können Backpropagation als Anwendung der Kettenregel erklären.
* [ ] Sie können für ein Mini-Netz einen Gradienten von Hand berechnen.
```

% TODO: Inhaltliche Unterabschnitte, zum Beispiel:
% ## Bergab in der Loss-Landschaft
% ## Die Learning Rate
% ## Backpropagation: die Kettenregel im Netz
% ## Ein Mini-Netz von Hand
% Übung T2 dieser Woche: NumPy-Mini-Netz mit vorgegebenem Gerüst
% (Forward implementieren, Backward nachvollziehen).

## Zusammenfassung und Ausblick

% TODO: Rückblick auf Kapitel 3; Ausblick auf Kapitel 4 — PyTorch
% automatisiert genau diese Rechnungen.
