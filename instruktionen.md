# Instruktionen für das Vorlesungsskript „Deep Learning für Maschinenbau"

Diese Instruktionen definieren verbindliche Konventionen für alle
MyST-Markdown-Quelldateien des Vorlesungsskripts. Abweichungen werden vor der
Umsetzung abgestimmt.

---

## 1. Projekt- und Dateistruktur

### Dateinamen

Dateinamen folgen dem Schema `chapterXX_secXX.md`, also zum Beispiel
`chapter01_sec01.md` für das erste Jupyter Notebook der ersten Woche.
Ein Kapitel entspricht einer Vorlesungswoche. Die letzte Sektion eines
Kapitels enthält die nummerierten Übungen. Ausnahme: Sektionen im
Studio-Format (siehe Abschnitt 2) enthalten ihre eigene nummerierte
Studio-Aufgabe direkt am Sektionsende, unabhängig davon, an welcher
Stelle im Kapitel die Sektion steht.

### YAML-Header

Jede Datei beginnt mit dem YAML-Header:

```yaml
---
kernelspec:
  name: python3
  display_name: 'Python 3'
---
```

### Seitenstruktur

Jede Sektion folgt dieser festen Abfolge:

1. **H1-Titel** (`# Titel der Sektion`)
2. **3-4 einleitende Sätze**: kein Inhaltsverzeichnis, kein „In diesem
   Kapitel lernen wir …", sondern ein konkretes Szenario oder eine
   Problemstellung, die Neugier weckt und den Abschnitt in den
   Gesamtworkflow einordnet.
3. **`## Lernziele`** mit der Lernziel-Box
4. Inhaltliche Unterabschnitte
5. **`## Aufgaben`** (nur in Sektionen im Studio-Format): die
   nummerierte Studio-Aufgabe für die Kleingruppenarbeit (siehe
   Abschnitt 2 und Abschnitt 6).
6. **`## Zusammenfassung und Ausblick`**

### Lernziele

Die Lernziele werden durch folgende Admonition hervorgehoben:

```{admonition} Lernziele
:class: attention
* [ ] Lernziel 1
* [ ] Lernziel 2
```

Lernziele werden **operationalisierbar** formuliert, nicht deklarativ: „Sie
können ein Training mit divergierender Verlustfunktion diagnostizieren" statt
„Sie kennen Optimierer". Die Lernziele aller Sektionen bilden zusammen die
Checkliste für die Rechnerklausur; jedes Lernziel muss prinzipiell prüfbar sein.

### Abschluss jeder Sektion

Jede Sektion endet mit:

```markdown
## Zusammenfassung und Ausblick

Kurzer Rückblick auf die Inhalte und Vorschau auf das nächste Kapitel.
```

---

## 2. Das Studio-Format: Zwei Exporte aus einer Quelldatei

Einige Sektionen werden im **Studio-Format** gehalten: ein Code-along
von etwa 25 Minuten im Termin, gefolgt von einer Kleingruppenarbeit von
etwa 15 Minuten an einer Studio-Aufgabe. Für dieses Format entstehen aus
derselben Quelldatei zwei unterschiedliche Jupyter-Notebook-Exporte für
zwei Zielgruppen; die Quelldatei selbst bleibt in jedem Fall die
Musterlösung mit gefüllten Zellen, wie sie auch in der HTML-Version auf
GitHub Pages erscheint.

### Die beiden Exporte

**Selbststudium-Notebook** (für Studierende, die nicht in Präsenz
teilnehmen): entspricht der Quelldatei nahezu unverändert. Alle
Kernzellen sind gefüllt (Musterlösung), alle Mini-Übungen bleiben
enthalten und dienen als Vorhersage- und Selbstkontrollmomente beim
eigenständigen Durcharbeiten.

**Präsenz-Notebook** (für den gemeinsamen Termin): Mini-Übungen werden
entfernt, denn die aktive Auseinandersetzung mit dem Stoff entsteht hier
live im Termin und anschließend in der Studio-Aufgabe. Stattdessen
erhalten ausgewählte Zeilen in den Kernzellen Lücken nach der
Lücken-Konvention (Abschnitt 5): Die Studierenden füllen sie gemeinsam
mit der Lehrperson im Termin.

Beide Exporte entstehen automatisiert aus derselben Quelldatei.

### Deckungsgleichheit von Lücken und Mini-Übungen

Lücken im Präsenz-Notebook und Mini-Übungen im Selbststudium-Notebook
sollen dieselben Schlüsselstellen einer Sektion treffen. Wer die
Mini-Übung zu einer Codezeile im Selbststudium löst, soll dieselbe Zeile
im Präsenztermin als Lücke aktiv ausfüllen. So üben beide Gruppen an
denselben Konzepten, nur in unterschiedlicher Sozialform.

### MyST-Kommentare zur Steuerung der Exporte

Die folgenden MyST-Kommentare (`% ...`) steuern die Erzeugung der
Exporte beziehungsweise dokumentieren Regieanweisungen für den Termin.
Sie erscheinen in keinem der drei Ausgabeformate (HTML,
Selbststudium-Notebook, Präsenz-Notebook) und dienen ausschließlich dem
Exportwerkzeug und der Lehrperson.

`% Lücke (Präsenz): <Beschreibung>` steht unmittelbar vor der
betroffenen Code-Zelle und markiert, welche Zeile(n) im Präsenz-Export
durch `# IHR CODE HIER` und `...` ersetzt werden:

````markdown
% Lücke (Präsenz): die Zeile `loss.backward()`

```{code-cell} python
...
loss.backward()
...
```
````

`% Live-Frage: <Frage> (Antwort: ...)` dokumentiert die mündliche
Vorhersagefrage, mit der die Lehrperson im Präsenztermin die dort
entfallende Mini-Übung ersetzt. Der Kommentar ist reine Regieanweisung
und wird nie exportiert.

`% Präsenz: <Anweisung>` dokumentiert sonstige Regieanweisungen zur
Zeitsteuerung im Termin, zum Beispiel welche Zellen im Termin nur
gezeigt und nicht gemeinsam eingetippt werden.

### Zeitbudget

Jede Sektion im Studio-Format dokumentiert ihr Gesamtzeitbudget als
Kommentar direkt nach dem H1-Titel:

```markdown
% Zeitbudget Präsenz (Code-along gesamt ca. 25 min): Video-1-Stoff ca. 9 min,
% Video-2-Stoff ca. 12 min, Video-3-Stoff ca. 4 min; danach Studio-Aufgabe.
```

### Gliederung nach Erklärvideos

Sektionen im Studio-Format werden zusätzlich mit Erklärvideos begleitet,
im Regelfall ein Video pro inhaltlichem H2-Abschnitt. Damit die Videos
in sich abgeschlossen bleiben und nicht zu kleinteilig werden, werden
inhaltliche Unterabschnitte auf maximal drei bis vier H2-Blöcke
konsolidiert, auch wenn das bedeutet, eng zusammenhängende Konzepte
(zum Beispiel eine Kernidee und ihr unmittelbarer Anwendungsfall) in
einem gemeinsamen Abschnitt statt in getrennten Abschnitten zu
behandeln. Jeder H2-Abschnitt erhält direkt unter der Überschrift einen
Kommentar mit Videonummer, geschätzter Videolänge und dem
Präsenz-Zeitbudget dieses Abschnitts:

```markdown
## Autograd: PyTorch rechnet die Gradienten

% Video 2 (ca. 10-12 min). Präsenz: ca. 12 min inklusive Lücken.
```

---

## 3. Sprache und Stil

### Grundsätze

- Alle Materialien sind auf **Deutsch**.
- Fachbegriffe auf Englisch werden beim ersten Auftreten erklärt.
- Englische Fachbegriffe, die im deutschen Satz als Nomen verwendet
  werden, werden großgeschrieben, auch wenn sie aus mehreren Wörtern
  bestehen. Das gilt auch für englische Nomen in Klammern als
  Übersetzungsangabe (zum Beispiel: `(englisch: "Loss")`).
  Wird ein englisches Wort hingegen als Prädikatsadjektiv verwendet,
  bleibt es klein (zum Beispiel: „das Modell ist overfittet").
- In den **Lernzielen** werden die Studierenden mit **Sie** angesprochen
  (zum Beispiel: „Sie können … erklären").
- In den **übrigen Lehrtexten** wird die gemeinsame Perspektive mit
  **wir** verwendet (zum Beispiel: „In diesem Kapitel haben wir …
  kennengelernt").
- Kein `d.h.` oder `z.B.` mitten im Satz: entweder ausschreiben oder in
  Klammern setzen.
- Vermeide Gedankenstriche.

### Deutsch vor Englisch bei etablierten Entsprechungen

Englische Fachbegriffe werden nur dort verwendet, wo im deutschsprachigen Raum
keine etablierte Entsprechung existiert (zum Beispiel: Batch Size). Existiert
eine den Studierenden vertraute deutsche Entsprechung, wird durchgängig diese
verwendet. Der englische Begriff wird beim ersten Auftreten einmalig als
Übersetzungsangabe in Klammern genannt (zum Beispiel: `(englisch: "Training
Loop")`), damit die Studierenden den Begriff in Dokumentation und Tutorials
wiedererkennen; der deutsche Begriff wird ins Glossar eingetragen.

Insbesondere gilt: **Schleife** statt "Loop", also for-Schleife,
innere Schleife und Trainingsschleife.

### Verbindliche Begriffsliste

Für zentrale Fachbegriffe gilt eine verbindliche Schreibweise, die in
`GLOSSAR.md` im Repository gepflegt wird. Neue Begriffe werden vor
der ersten Verwendung ins Glossar eingetragen. Das Glossar wird als
Anhang in das Skript eingebunden.

### Rhetorische Fragen

- Rhetorische Fragen sind ausdrücklich erwünscht, um Aufmerksamkeit zu
  lenken und Spannung aufzubauen.
- Rhetorische Fragen werden kursiv gesetzt:

*Aber warum divergiert das Training, sobald wir die Learning Rate erhöhen?*

---

## 4. Didaktische Prinzipien

### Prinzip: Erst Beispiel, dann abstrakt

Jedes neue Konzept wird nach folgendem Dreischritt eingeführt:

1. **Konkretes Beispiel zuerst:** Wir beschreiben ein praktisches
   Problem oder Phänomen in Alltagssprache (zum Beispiel: das Netz mit
   10.000 Gewichten — an welchem sollen wir drehen, und in welche
   Richtung?).
2. **Informelle Beschreibung:** Wir benennen das Muster oder den Grund
   dafür noch ohne formalen Begriff (wir brauchen die Empfindlichkeit
   der Loss bezüglich jedes einzelnen Gewichts).
3. **Begriff oder Erklärung:** Erst jetzt führen wir den Fachbegriff
   oder die technische Erklärung ein (Gradient, Backpropagation).

### Brücken zur Ingenieurwelt

Wo immer möglich, knüpfen Erklärungen an Konzepte an, die
Maschinenbau-Studierende aus ihrem Studium oder ihren Alltag kennen (zum
Beispiel: Gradient ↔ Sensitivitätsanalyse, Regularisierung ↔ Sicherheitsfaktor,
iterative Optimierung ↔ Newton-Verfahren). Solche Brücken werden im Fließtext
gesetzt, nicht in eigene Boxen ausgelagert.

---

## 5. Ausführbare Code-Zellen

### Grundform

```{code-cell} python
print("Example")
```

### Sprache in Code-Zellen

Für die Variablen werden englische Begriffe verwendet.

### Laufzeitregel (verbindlich)

Jede Code-Zelle muss auf CPU in **unter etwa 30 Sekunden** ausführbar
sein. Trainings-Zellen im Skript sind deshalb auf wenige Epochen und
kleine Modelle dimensioniert. „Echte" Trainingsverläufe (viele Epochen,
große Modelle) werden nicht im Build ausgeführt, sondern als vorab
erzeugte Abbildungen oder gespeicherte Logs eingebunden. Für den
MyST-Build wird der Jupyter-Cache aktiviert, damit unveränderte Zellen
nicht neu ausgeführt werden.

### Reproduzierbarkeit

Jede Zelle, die Zufallszahlen verwendet (Initialisierung, Shuffling,
Splits), setzt zu Beginn einen Seed:

```python
import torch
torch.manual_seed(42)
```

### Visualisierung

Plots, die aus Code-Zellen entstehen (Lernkurven, Datenexploration,
Vorhersagen), werden mit **Plotly** erzeugt — interaktive Lernkurven
sind ein Mehrwert des HTML-Skripts. Statische Konzeptabbildungen
entstehen dagegen mit TikZ (siehe Abschnitt 9).

### Lücken-Konvention

In den Studierenden-Notebooks (jupytext-Export für Moodle) werden
Lücken einheitlich markiert:

```python
# IHR CODE HIER
loss = ...
```

Der Marker `# IHR CODE HIER` und der Platzhalter `...` werden
ausschließlich in dieser Form verwendet, damit Quelldateien,
Lückennotebooks und Musterlösungen automatisiert konsistent gehalten
werden können. Die HTML-Version auf GitHub Pages zeigt immer die
gefüllten Zellen (Musterlösung). Im Präsenz-Export des Studio-Formats
(siehe Abschnitt 2) legt der Kommentar `% Lücke (Präsenz): ...` fest,
welche Zeile einer ansonsten gefüllten Kernzelle zur Lücke wird.

---

## 6. Übungen

### Mini-Übungen (innerhalb der Theorieteile)

```{admonition} Mini-Übung
:class: tip
Aufgabentext hier.
```

```{code-cell} python
# Code-Zelle

```

````{admonition} Lösung
:class: tip
:class: dropdown
```python
# Musterlösung hier
```
````

Mini-Übungen dienen im Code-along-Format als Vorhersage- und
Selbstkontrollmomente: bevorzugt Aufgaben des Typs „Was wird diese
Zelle ausgeben?" oder „Ergänzen Sie die fehlende Zeile" — kurz genug
für 2-3 Minuten im Termin. In Sektionen im Studio-Format erscheinen
Mini-Übungen ausschließlich im Selbststudium-Export (siehe Abschnitt 2);
im Präsenz-Export entfallen sie und werden durch eine mündliche
Live-Frage der Lehrperson ersetzt.

### Nummerierte Übungen (Schwierigkeitsgrad mit Sternen)

Schwierigkeitsgrade:

```{admonition} Übung X.Y (✩)             ← einfach
```

```{admonition} Übung X.Y (✩✩)            ← mittel
```

```{admonition} Übung X.Y (✩✩✩)           ← schwer
```

```{admonition} Übung X.Y (Mini-Projekt)  ← umfangreich
```

**Kalibrierung:** ✩✩ entspricht Klausurniveau. Dies wird den
Studierenden im Vorwort und in Woche 1 explizit mitgeteilt.
Übungen der Kategorie „Mini-Projekt" bereiten gezielt auf die
Projektphase vor (eigenständiger Workflow auf einem neuen Datensatz).

### Studio-Aufgabe (Kleingruppenübung im Studio-Format)

In Sektionen im Studio-Format (siehe Abschnitt 2) steht die nummerierte Übung
nicht in der letzten Sektion des Kapitels, sondern direkt am Ende der eigenen
Sektion unter der Überschrift `## Aufgaben`, kurz vor `## Zusammenfassung und
Ausblick`. Sie erhält eine Sternebewertung wie jede andere nummerierte Übung bei
(zum Beispiel Übung (✩✩)), da die Sterne weiterhin die Klausurkalibrierung
transportieren, aber keine Nummer.

Die Studio-Aufgabe ist für die Bearbeitung in Kleingruppen im Anschluss
an den Code-along vorgesehen (etwa 15 Minuten) und verwendet dieselbe
Lücken-Konvention wie die Kernzellen im Präsenz-Notebook: Das
Aufgabengerüst enthält `# IHR CODE HIER`-Markierungen genau an den
Stellen, die den im Code-along behandelten Bausteinen entsprechen. Die
Teilaufgaben spiegeln die Reihenfolge dieser Bausteine, damit die
Kleingruppen dem Termin unmittelbar folgen können, statt neu
orientieren zu müssen.

### Backtick-Regel für Admonitions

Admonition-Blöcke werden grundsätzlich mit **drei Backticks** (` ``` `)
geöffnet und geschlossen. **Vier Backticks** (` ```` `) sind nur dann
erforderlich, wenn der Block einen eingebetteten Fenced Code Block
enthält (` ```python ... ``` `), da MyST sonst den inneren Block als
Abschluss des äußeren interpretiert.

Drei Backticks (Normalfall, kein eingebetteter Code):

```{admonition} Übung X.Y (✩)
:class: tip
Aufgabentext hier.
```

```{admonition} Lösung
:class: tip
:class: dropdown
Lösungstext hier.
```

Vier Backticks (nur wenn die Lösung einen ```python-Block enthält):

```{admonition} Übung X.Y (✩✩)
:class: tip
Aufgabentext hier.
```

````{admonition} Lösung
:class: tip
:class: dropdown
```python
# Musterlösung hier
```
````

---

## 7. Wiederkehrende Hinweis-Boxen

### Rechenumgebung (GPU-Hinweis)

Notebooks, die sinnvoll nur mit GPU laufen (ab dem CNN-Kapitel),
erhalten direkt nach den Lernzielen eine einheitliche Box:

```{admonition} Rechenumgebung: GPU empfohlen
:class: warning
Dieses Notebook enthält Trainingsläufe, die auf CPU sehr lange dauern.
Verwenden Sie Google Colab (Laufzeittyp: GPU) oder den JupyterHub der
Hochschule. Eine Anleitung finden Sie in Kapitel 1.
```

Die Abstufungen „GPU empfohlen" und „GPU erforderlich" werden
einheitlich verwendet; weitere Varianten sind nicht zulässig.

### Reproduzierbarkeits-Hinweis

Beim ersten Trainingslauf eines Kapitels wird einmalig folgender
Standardbaustein eingefügt (Wortlaut nicht variieren):

```{admonition} Hinweis: Ihre Zahlen können abweichen
:class: warning
Wir setzen in allen Notebooks einen Seed, damit die Ergebnisse
reproduzierbar sind. Training auf einer GPU ist dennoch nie perfekt
deterministisch. Wenn Ihre Loss-Werte geringfügig von den hier
gezeigten abweichen, ist das normal und kein Fehler.
```

---

## 8. Medien und Tabellen

### Eingebettete YouTube-Videos

```{dropdown} Video "Titel" von Quelle
<iframe width="560" height="315" src="https://www.youtube.com/embed/VIDEO_ID"
title="YouTube video player" frameborder="0" allow="accelerometer; autoplay;
clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe>
```

Englischsprachige Videos sind zulässig und im Dropdown-Titel als
solche zu kennzeichnen (zum Beispiel: `Video (EN) "Backpropagation" von
3Blue1Brown`). Im Vorwort des Skripts wird darauf
hingewiesen, dass zentrale Erklärvideos zu Deep Learning überwiegend
auf Englisch verfügbar sind.

### Markdown-Tabellen

Tabellen werden im **compact-Stil** gesetzt: In der Trennzeile steht
auf beiden Seiten der Striche je ein Leerzeichen. Die Variante ohne
Leerzeichen (`|---|`) ist nicht erlaubt, da sie die Linter-Warnung
MD060 auslöst.

```markdown
| Spalte 1 | Spalte 2 | Spalte 3 |
| -------- | -------- | -------- |
| Inhalt   | Inhalt   | Inhalt   |
```

---

## 9. TikZ-Abbildungen

Alle Abbildungen werden als eigenständige `standalone`-Dokumente in
LaTeX erstellt, nach SVG exportiert und über eine `{figure}`-Direktive
in die MyST-Markdown-Quelldatei eingebunden. Die Kompilierung erfolgt
mit `lualatex`.

### Präambel-Vorlage

Jede TikZ-Datei verwendet ausnahmslos die folgende Präambel:

```latex
\documentclass[11pt]{standalone}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage[no-math]{fontspec}
\usepackage{unicode-math}
\setmainfont{Libertinus Sans}
\setsansfont{Libertinus Sans}
\setmathfont{Libertinus Math}
\usepackage{pgf,xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, backgrounds}
\usepackage{pgfplots}
\pgfplotsset{compat=newest}
\usepackage{pifont}
```

`pgfplots` wird in jeder Datei eingebunden, auch wenn die Abbildung nur
reines TikZ verwendet, damit die Präambel über alle Dateien identisch
bleibt. Die Präambel darf bei Bedarf erweitert werden.

### Schrift

Die Abbildungen verwenden **Libertinus Sans** als Textschrift und
**Libertinus Math** als Mathematikschrift. Libertinus Sans passt als
serifenlose Schrift zum Standard-Theme von Jupyter Book; Libertinus
Math stellt sicher, dass Formeln in Abbildungen korrekt gesetzt werden.
Das Paket `\usepackage{libertinus}` wird nicht verwendet, da die
Schriften explizit über `fontspec` und `unicode-math` konfiguriert
werden. Keine anderen Schriftfamilien verwenden (insbesondere nicht
TeX Gyre Heros oder Computer Modern).

### Farbpalette

Alle sieben Farben werden in jeder Datei definiert, auch wenn nur eine
Teilmenge verwendet wird:

```latex
\definecolor{my_darkgray}{HTML}{484949}
\definecolor{my_lightgray}{HTML}{F3F4F4}
\definecolor{my_darkblue}{HTML}{005A94}
\definecolor{my_lightblue}{HTML}{CCDEE9}
\definecolor{my_yellow}{HTML}{FFEC7F}
\definecolor{my_red}{HTML}{E60000}
\definecolor{my_orange}{HTML}{E87846}
```

Keine ad-hoc-Farbnamen einführen. Alle Farben kommen ausnahmslos aus
dieser Palette.

**Zweifarbige Abbildungen (Standardfall):** `my_darkblue` für das
primäre Element (Konturlinie, Hauptkurve, dominante Region) und
`my_lightblue` für das sekundäre Element (Füllung, Hintergrundregion,
zweite Kurve).

**Dreifarbige Funktionsgraphen:** `my_darkblue` für die erste Kurve,
`my_red` für die zweite, `my_orange` für die dritte. `my_yellow` nicht
für Kurvenlinien verwenden (zu wenig Kontrast). `my_yellow` ist für
hervorgehobene Füllregionen reserviert.

**Text und Annotationen:** `my_darkgray` für sekundäre Labels.
`my_darkblue` für Labels, die mathematische oder technische Objekte
bezeichnen.

### Hintergrundpanel

Jede Abbildung hat ein explizites Hintergrundpanel mit `my_lightgray`.
Dies gewährleistet korrekte Darstellung in hellen und dunklen
Browser-Themes:

```latex
\begin{tikzpicture}[
    show background rectangle,
    background rectangle/.style={fill=my_lightgray, rounded corners=8pt},
    inner frame sep=0.8cm
]
```

`inner frame sep=0.3cm` nur für breite, flache Abbildungen (zum
Beispiel Zahlenstrahlen). In allen anderen Fällen `0.8cm`.

### Achsenstil für Funktionsgraphen (pgfplots)

pgfplots wird nur für statische Konzeptgraphen verwendet
(Aktivierungsfunktionen, schematische Loss-Landschaften). Lernkurven
und Datenplots entstehen aus Code-Zellen mit Plotly (siehe Abschnitt 5).

```latex
\begin{axis}[
    axis lines = center,
    xlabel = {$x$},
    ylabel = {$y$},
    grid = both,
    axis equal,
    axis line style={thick},
]
```

`axis equal` nur weglassen, wenn das natürliche Seitenverhältnis der
Funktion den Graphen unleserlich macht. Die Ausnahme mit einem
Kommentar dokumentieren. Alle Kurven:

```latex
\addplot[draw=my_darkblue, samples=300, ultra thick, domain=a:b]{ ... };
```

### Legenden

Bei mehr als einer Kurve eine Legende einfügen:

```latex
legend pos=north west,
legend style={font=\small},
legend cell align=left,
```

`\addlegendentry{...}` direkt nach dem jeweiligen `\addplot`-Aufruf.

### Schematische Diagramme und technische Abbildungen

Für Diagramme ohne Koordinatenachsen (annotierte Skizzen,
Konzeptdiagramme, Mehrpanel-Vergleiche) gelten folgende Ergänzungen.

**Annotationspfeil:** Beschriftungspfeile verwenden einheitlich den
Stealth-Pfeilkopf in `my_darkgray`. Die Stil-Definition gehört in das
optionale Argument der `tikzpicture`-Umgebung:

```latex
annotation/.style={draw=my_darkgray, thick, -{Stealth[length=7pt, width=5pt]}}
```

Labels neben Annotationspfeilen: `font=\small, text=my_darkgray`.

**Mehrpanel-Layout:** Panels werden mit `\begin{scope}[xshift=...]`
nebeneinander gesetzt. Der Abstand zwischen zwei Panels beträgt
mindestens `1cm`. Vertikale Trennlinien zwischen Panels sind nicht
erforderlich.

**Panel-Untertitel:** Jedes Panel erhält einen Titel direkt unterhalb
des Inhalts, zentriert:

```latex
\node[font=\small\bfseries, text=my_darkgray, anchor=north] at (...) {...};
```

**Hervorgehobene Elemente:**

- Fläche oder Füllregion: `fill=my_yellow`
- Linienelement (Kante, Achse): `draw=my_orange, ultra thick`
- Knotenpunkt: `fill=my_orange` mit weißem Rand
  (`draw=my_lightgray, thick`)

**Bewertungssymbole:** ✓ und ✗ werden über das Paket `pifont` gesetzt
(bereits in der Präambel). `\ding{51}` für ✓, `\ding{55}` für ✗.
Unicode-Zeichen dürfen in Kommentaren erscheinen, aber nicht im
ausführbaren LaTeX-Code. `font=\large\bfseries`. ✓ in `my_darkblue`,
✗ in `my_red`.

### Netzarchitektur-Diagramme (neu für Deep Learning)

Für Architekturdiagramme (Netzgraphen, Schichtdarstellungen,
Faltungsoperationen) gelten einheitliche Stile, die in jeder
betreffenden Datei in das optionale Argument der
`tikzpicture`-Umgebung aufgenommen werden:

```latex
neuron/.style={circle, draw=my_darkblue, thick, fill=my_lightblue,
    minimum size=7mm, inner sep=0pt},
neuron highlight/.style={neuron, fill=my_orange,
    draw=my_lightgray, thick},
layer/.style={rectangle, draw=my_darkblue, thick, fill=my_lightblue,
    rounded corners=2pt, minimum width=12mm, minimum height=28mm},
connection/.style={draw=my_darkgray, thin, -{Stealth[length=5pt, width=4pt]}},
connection highlight/.style={connection, draw=my_orange, ultra thick}
```

Konventionen:

- **Neuronen** als Kreise (`neuron`), das jeweils diskutierte Neuron
  mit `neuron highlight`.
- **Schichten** in kompakten Architekturübersichten als Blöcke
  (`layer`) statt einzelner Neuronen, sobald ein Netz mehr als drei
  Schichten hat.
- **Verbindungen/Datenfluss** mit `connection`; der im Text
  besprochene Pfad mit `connection highlight`.
- **Hervorgehobene Regionen** (zum Beispiel das rezeptive Feld einer
  Faltung, ein Filterfenster): `fill=my_yellow`.
- Datenfluss verläuft in allen Abbildungen einheitlich **von links
  nach rechts**.
