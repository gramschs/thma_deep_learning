---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Das erste lernende Modell: lineare Regression

% TODO: 3–4 einleitende Sätze — konkretes Szenario.
% Vorschlag: Aus Prüfstandsdaten soll der Energieverbrauch einer
% Anlage in Abhängigkeit von der Drehzahl vorhergesagt werden — wir
% lassen die Parameter der Geradengleichung aus den Daten "lernen".
% Didaktischer Kern: Das Denkmuster Modell – Loss – Optimierung wird
% hier eingeführt und überträgt sich 1:1 auf neuronale Netze (Woche 3).

```{admonition} Lernziele
:class: attention
* [ ] Sie können die Bestandteile eines lernenden Modells benennen und an einem Beispiel erklären: Modell, Parameter, Loss, Optimierung.
* [ ] Sie können den Mean Squared Error als Loss erklären und für ein kleines Beispiel von Hand berechnen.
* [ ] Sie können eine lineare Regression an einen Datensatz anpassen und die gelernten Parameter fachlich interpretieren.
* [ ] Sie können begründen, warum das Anpassen der Parameter ein Optimierungsproblem ist.
```

% TODO: Inhaltliche Unterabschnitte, zum Beispiel:
% ## Ein Modell mit Stellschrauben
% ## Wie gut ist das Modell? Die Loss
% ## Die beste Gerade finden
% ## Mehr als eine Einflussgröße
% Erst Beispiel, dann abstrakt: mit konkreten Prüfstandsdaten starten,
% MSE erst informell ("mittlerer quadratischer Fehler"), dann formal.

## Zusammenfassung und Ausblick

% TODO: Rückblick; Ausblick auf die Frage, ob das Modell auch auf
% neuen Daten funktioniert (Generalisierung, nächste Sektion).
