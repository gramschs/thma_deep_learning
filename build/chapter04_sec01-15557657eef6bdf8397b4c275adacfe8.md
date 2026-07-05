---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# 4.1 PyTorch: Tensoren und Autograd

% Hinweise zur Datei:
% - Diese Quelldatei ist die Musterlösung: alle Code-Zellen sind gefüllt
%   (so erscheint sie in der HTML-Version auf GitHub Pages).
% - Präsenz-Export: Mini-Übungen entfernen; die mit "% Lücke (Präsenz):"
%   markierten Zeilen durch `# IHR CODE HIER` und `...` ersetzen.
% - Selbststudium-Export: Mini-Übungen bleiben enthalten, keine Lücken.
% - "% Live-Frage:" = mündliche Vorhersagefrage für den Präsenztermin
%   (ersetzt dort die entfallende Mini-Übung).
% TODO Glossar: Tensor, Autograd, Berechnungsgraph (Computational Graph),
% Gradientenakkumulation vor Veröffentlichung in GLOSSAR.md eintragen.

In Kapitel 3 haben wir die Gradienten unseres Mini-Netzes von Hand berechnet:
vier Gewichte, eine halbe Seite Kettenregel. Ein Netz, das Schweißnähte auf
Fotos prüft, hat mehrere Millionen Gewichte, und nach jeder Änderung der
Architektur müssten wir sämtliche Ableitungen neu herleiten. *Wer soll das
rechnen?* Genau hier setzt PyTorch an: Die Bibliothek speichert unsere Daten
als Tensoren und leitet jede Rechnung, die wir damit ausführen, vollautomatisch
ab. In dieser Sektion lernen wir beide Bausteine kennen und prüfen am Beispiel
aus Kapitel 3 nach, dass PyTorch tatsächlich dieselben Gradienten liefert wie
unsere Handrechnung.

## Lernziele

```{admonition} Lernziele
:class: attention
* [ ] Sie können PyTorch-Tensoren erzeugen und zwischen NumPy-Arrays und
  Tensoren konvertieren.
* [ ] Sie können erklären, was `requires_grad` bewirkt und wie PyTorch den
  Berechnungsgraphen aufbaut.
* [ ] Sie können mit `backward()` Gradienten berechnen und das Ergebnis mit
  einer Handrechnung aus Kapitel 3 abgleichen.
* [ ] Sie können erklären, welche Arbeit Autograd uns abnimmt und welche nicht.
```

```{admonition} Hinweis: Ihre Zahlen können abweichen
:class: warning
Wir setzen in allen Notebooks einen Seed, damit die Ergebnisse
reproduzierbar sind. Training auf einer GPU ist dennoch nie perfekt
deterministisch. Wenn Ihre Loss-Werte geringfügig von den hier
gezeigten abweichen, ist das normal und kein Fehler.
```

## Tensoren: Daten in PyTorch

% Video 1 (ca. 7-8 min): Tensoren erzeugen, shape/dtype, NumPy-Brücke

Aus der Technischen Mechanik kennen wir Vektoren und Matrizen: Ein Kraftvektor
hat drei Komponenten, eine Steifigkeitsmatrix ist ein rechteckiges Zahlenschema.
Deep-Learning-Daten sprengen dieses Raster schnell. Ein Farbbild hat Höhe,
Breite und drei Farbkanäle, ein Stapel aus 32 Farbbildern sogar vier
Dimensionen. PyTorch fasst deshalb alles unter einem einzigen Oberbegriff
zusammen: dem **Tensor**, einem Zahlenschema mit beliebig vielen Dimensionen.
Ein Skalar ist ein Tensor mit null Dimensionen, ein Vektor einer mit einer
Dimension, eine Matrix einer mit zwei Dimensionen, und so weiter.

Als durchgehendes Beispiel dieser Sektion begleitet uns eine Schraubenfeder aus
dem Prüflabor: Wir lenken sie unterschiedlich weit aus und messen die
Rückstellkraft. Vier Messwerte der Kraft (in Newton) speichern wir als unseren
ersten Tensor. Wie bei Pandas und NumPy importieren wir zuerst das Modul; die
übliche Abkürzung für PyTorch ist schlicht `torch`.

```{code-cell} python
import torch

kraefte = torch.tensor([2.9, 5.1, 6.6, 9.0])
print(kraefte)
```

Die Ausgabe sieht fast aus wie eine Liste, trägt aber das Etikett `tensor`.
Was ein Tensor über die reinen Zahlen hinaus weiß, verraten uns seine
Attribute, allen voran `shape` (die Größe entlang jeder Dimension) und
`dtype` (der gemeinsame Datentyp aller Elemente):

```{code-cell} python
print(kraefte.shape)
print(kraefte.dtype)
```

`torch.Size([4])` bedeutet: eine Dimension mit vier Einträgen, also ein
Vektor der Länge 4. Der Datentyp `float32` ist eine 32-Bit-Kommazahl und der
Standard im Deep Learning: Sie braucht nur halb so viel Speicher wie die aus
NumPy gewohnte 64-Bit-Kommazahl, rechnet auf GPUs deutlich schneller, und die
geringere Genauigkeit spielt beim Training neuronaler Netze praktisch keine
Rolle. Wie bei einer Pandas-Series gilt: Ein Tensor erzwingt einen gemeinsamen
Datentyp für alle Elemente.

Tensoren mit mehr oder weniger Dimensionen erzeugen wir nach demselben Muster.
Ein Skalar entsteht aus einer einzelnen Zahl, eine Matrix aus einer Liste von
Listen:

```{code-cell} python
temperatur = torch.tensor(21.5)
messmatrix = torch.tensor([[1.0, 2.9], [2.0, 5.1], [3.0, 6.6], [4.0, 9.0]])
print(temperatur.shape)
print(messmatrix.shape)
```

Der Skalar hat die leere Shape `torch.Size([])`, also null Dimensionen. Die
Matrix hat die Shape `[4, 2]`: vier Zeilen (Messungen) und zwei Spalten
(Auslenkung und Kraft). Ein kurzer Blick auf `shape` ist im Alltag das
wichtigste Diagnosewerkzeug überhaupt, denn die meisten Fehlermeldungen in
PyTorch sind am Ende Shape-Konflikte.

% Live-Frage: Welche Shape hätte ein Stapel aus 32 Graustufenbildern
% mit je 28 x 28 Pixeln? (Antwort: [32, 28, 28])

```{admonition} Mini-Übung
:class: tip
Ordnen Sie zu: Wie viele Dimensionen hat der passende Tensor, und wie könnte
seine Shape lauten?

1. Die aktuelle Temperatur eines einzelnen Sensors.
2. Eine Messreihe aus 500 Kraftwerten.
3. Ein Graustufenbild mit 28 x 28 Pixeln.
4. Ein Stapel aus 32 solchen Graustufenbildern.
```

```{admonition} Lösung
:class: tip
:class: dropdown
1. Null Dimensionen (Skalar), Shape `[]`.
2. Eine Dimension (Vektor), Shape `[500]`.
3. Zwei Dimensionen (Matrix), Shape `[28, 28]`.
4. Drei Dimensionen, Shape `[32, 28, 28]`. Die neue, vorangestellte Dimension
   zählt die Bilder im Stapel. Genau in dieser Form werden wir später
   Trainingsdaten portionsweise durch ein Netz schicken.
```

Für Gewichte, die später zufällig initialisiert werden, brauchen wir Tensoren
aus Zufallszahlen. Damit die Ergebnisse reproduzierbar bleiben, setzen wir wie
immer zuerst einen Seed:

```{code-cell} python
torch.manual_seed(42)
zufallsgewichte = torch.randn(3)
print(zufallsgewichte)
```

`torch.randn` zieht standardnormalverteilte Zufallszahlen, hier drei Stück.
Daneben gibt es unter anderem `torch.zeros` und `torch.ones` für Tensoren
voller Nullen beziehungsweise Einsen; das Muster ist immer dasselbe, das
Argument bestimmt die Shape.

In der Praxis liegen Messdaten selten als handgetippte Listen vor, sondern
kommen als NumPy-Arrays aus der Messdatenerfassung oder aus Pandas. Die Brücke
zwischen beiden Welten ist kurz: `torch.from_numpy` macht aus einem Array einen
Tensor, die Methode `.numpy()` führt zurück.

```{code-cell} python
import numpy as np

auslenkung_np = np.array([1.0, 2.0, 3.0, 4.0])
auslenkung = torch.from_numpy(auslenkung_np)
print(auslenkung)
print(auslenkung.dtype)
```

Hier lohnt der zweite Blick auf den Datentyp: `float64` statt `float32`, denn
NumPy rechnet standardmäßig mit 64 Bit und `torch.from_numpy` übernimmt den
Datentyp unverändert. Für das Training konvertieren wir deshalb üblicherweise
mit `.float()` in das PyTorch-Standardformat:

```{code-cell} python
auslenkung = torch.from_numpy(auslenkung_np).float()
print(auslenkung.dtype)
```

% Live-Frage: Wir ändern nach from_numpy das NumPy-Array. Ändert sich der
% Tensor mit? (Antwort: ja, geteilter Speicher; .float() erzeugt eine Kopie)

````{admonition} Mini-Übung
:class: tip
Eine Kollegin erzeugt einen Tensor mit `torch.from_numpy` und ändert
anschließend das NumPy-Array:

```python
messung_np = np.array([1.0, 2.0, 3.0])
messung = torch.from_numpy(messung_np)
messung_np[0] = 99.0
print(messung)
```

Was gibt die letzte Zeile aus? Stellen Sie zuerst eine Vermutung auf und
prüfen Sie sie dann in der Code-Zelle.
````

```{code-cell} python
# Code-Zelle
```

```{admonition} Lösung
:class: tip
:class: dropdown
Die Ausgabe ist `tensor([99., 2., 3.], dtype=torch.float64)`. Der Tensor hat
sich mitgeändert, denn `torch.from_numpy` kopiert die Daten nicht, sondern
teilt sich den Speicher mit dem NumPy-Array. Das ist schnell und spart
Speicher, kann aber überraschen. Wer eine unabhängige Kopie braucht, verwendet
`torch.tensor(messung_np)` oder hängt wie oben `.float()` an, denn die
Typkonvertierung erzeugt ebenfalls eine Kopie.
```

## Autograd: PyTorch rechnet die Gradienten

% Video 2 (ca. 10-12 min): requires_grad, Berechnungsgraph, backward(),
% Abgleich mit der Handrechnung aus Kapitel 3

Zurück zu unserem Problem aus der Einleitung. In Kapitel 3 haben wir für jedes
Gewicht einzeln die Kettenregel aufgeschrieben, um zu bestimmen, wie empfindlich
die Loss auf eine kleine Änderung dieses Gewichts reagiert. Bei vier Gewichten
war das mühsam, bei Millionen ist es aussichtslos. *Wie sagen wir PyTorch, dass
es diese Arbeit übernehmen soll?* Wir beginnen mit dem einfachsten Beispiel, das
wir noch im Kopf ableiten können: die Funktion $y = x^2$ an der Stelle $x = 3$.
Die Ableitung kennen wir aus dem ersten Semester, $y' = 2x$, also erwarten wir
den Wert 6.

Der Schlüssel ist ein einziges zusätzliches Argument beim Erzeugen des Tensors:
`requires_grad=True`. Damit melden wir die Variable bei PyTorch an: Beobachte
alles, was mit diesem Tensor gerechnet wird.

% Lücke (Präsenz): das Argument `requires_grad=True`

```{code-cell} python
x = torch.tensor(3.0, requires_grad=True)
y = x**2
print(y)
```

Die Ausgabe enthält neben dem erwarteten Wert `9.` einen unscheinbaren Zusatz:
`grad_fn=<PowBackward0>`. Er verrät, dass PyTorch beim Quadrieren nicht nur das
Ergebnis berechnet, sondern sich auch gemerkt hat, *welche* Operation das
Ergebnis erzeugt hat (eine Potenz, englisch: "Power"). Jede weitere Rechnung
mit `x` oder `y` würde ebenso protokolliert. Aus diesen Protokolleinträgen
entsteht Schritt für Schritt eine Kette, genauer ein Netz von Operationen: der
**Berechnungsgraph** (englisch: "Computational Graph"). Er hält fest, wie der
Ausgabewert aus den Eingangsgrößen entstanden ist, und ist damit genau die
Struktur, an der die Kettenregel entlanglaufen kann. Dieses automatische
Ableiten entlang des Graphen heißt in PyTorch **Autograd**.

% TODO Abbildung: Berechnungsgraph des Mini-Netzes aus Kapitel 3 als TikZ
% (Netzarchitektur-Stile aus den Instruktionen, Datenfluss von links nach
% rechts, Knoten: x, w1, b1, z, sigmoid, h, w2, b2, y_dach, Loss;
% den Pfad der Kettenregel zu w1 mit connection highlight hervorheben).
% Einbindung dann über:
% ```{figure} ../img/chapter04/berechnungsgraph_mininetz.svg
% :name: fig-berechnungsgraph
% Der Berechnungsgraph des Mini-Netzes aus Kapitel 3.
% ```

Den Anstoß zur eigentlichen Gradientenberechnung gibt die Methode
`backward()`: Sie läuft den Berechnungsgraphen von hinten nach vorn ab (daher
der Name) und wendet dabei an jedem Knoten die Kettenregel an. Das Ergebnis
landet im Attribut `.grad` der beobachteten Variablen:

% Lücke (Präsenz): die Zeile `y.backward()`

```{code-cell} python
y.backward()
print(x.grad)
```

Dort steht `tensor(6.)`, exakt der Wert unserer Kopfrechnung $2x = 6$. Halten
wir den Dreiklang fest, denn er kehrt in jedem Training wieder:
`requires_grad=True` meldet Variablen an, die Vorwärtsrechnung baut den
Berechnungsgraphen auf, `backward()` füllt die `.grad`-Attribute.

```{admonition} Mini-Übung
:class: tip
Berechnen Sie mit Autograd die Ableitung der Funktion $f(x) = x^3 + 2x$ an der
Stelle $x = 2$. Rechnen Sie zuerst von Hand nach, welcher Wert herauskommen
muss, und prüfen Sie dann mit PyTorch.
```

```{code-cell} python
# Code-Zelle
```

````{admonition} Lösung
:class: tip
:class: dropdown
Von Hand: $f'(x) = 3x^2 + 2$, also $f'(2) = 14$.

```python
x = torch.tensor(2.0, requires_grad=True)
f = x**3 + 2*x
f.backward()
print(x.grad)
```

Die Ausgabe ist `tensor(14.)` und bestätigt die Handrechnung.
````

Jetzt zum eigentlichen Härtetest: unserem Mini-Netz aus Kapitel 3. Zur
Erinnerung: Ein Eingang $x = 2$ läuft in ein verstecktes Neuron mit der
Sigmoid-Aktivierung, dessen Ausgang $h$ in ein lineares Ausgabeneuron. Die vier
Parameter waren $w_1 = 0{,}5$, $b_1 = 0$, $w_2 = -0{,}3$ und $b_2 = 0{,}1$, der
Zielwert $y = 1$, die Loss der quadratische Fehler. Unsere Handrechnung über
die Kettenregel hatte folgende Gradienten ergeben:

% TODO: Zahlenwerte und Notation mit der endgültigen Fassung von Kapitel 3
% abgleichen (hier gerundet auf vier Nachkommastellen).

| Parameter | Gradient (Handrechnung) |
| --------- | ----------------------- |
| $w_1$     | 0.2641                  |
| $b_1$     | 0.1320                  |
| $w_2$     | -1.6366                 |
| $b_2$     | -2.2386                 |

Dieselbe Rechnung in PyTorch: Wir legen die vier Parameter mit
`requires_grad=True` an, schreiben die Vorwärtsrechnung als ganz gewöhnliche
Python-Zeilen und überlassen den Rest Autograd.

% Lücke (Präsenz): die vier `requires_grad=True`-Argumente sowie die Zeile
% `loss.backward()`

```{code-cell} python
x = torch.tensor(2.0)
y_wahr = torch.tensor(1.0)

w1 = torch.tensor(0.5, requires_grad=True)
b1 = torch.tensor(0.0, requires_grad=True)
w2 = torch.tensor(-0.3, requires_grad=True)
b2 = torch.tensor(0.1, requires_grad=True)

h = torch.sigmoid(w1 * x + b1)
y_dach = w2 * h + b2
loss = (y_dach - y_wahr)**2
print(f'Loss: {loss.item():.4f}')

loss.backward()
print(f'Gradient w1: {w1.grad.item():.4f}')
print(f'Gradient b1: {b1.grad.item():.4f}')
print(f'Gradient w2: {w2.grad.item():.4f}')
print(f'Gradient b2: {b2.grad.item():.4f}')
```

Alle vier Werte stimmen Ziffer für Ziffer mit der Tabelle überein. Die halbe
Seite Kettenregel aus Kapitel 3 steckt in einer einzigen Zeile:
`loss.backward()`. Die Methode `.item()` haben wir hier nebenbei
kennengelernt: Sie holt aus einem Tensor mit genau einem Element die gewöhnliche
Python-Zahl heraus, was die formatierte Ausgabe erleichtert. Und das Beste:
Dieses Vorgehen skaliert. Ob vier Parameter oder vier Millionen, ob unsere
kleine Formel oder ein Netz mit fünfzig Schichten, der Ablauf bleibt identisch,
nur der Berechnungsgraph wird größer.

## Was Autograd übernimmt und was nicht

% Video 3 (ca. 4-5 min): Gradientenakkumulation, torch.no_grad(),
% Arbeitsteilung Mensch/Autograd; direkte Vorbereitung der Studio-Aufgabe

Autograd nimmt uns das Ableiten ab, aber nicht das Denken. Zwei Eigenheiten
müssen wir kennen, bevor wir in der Studio-Aufgabe zum ersten Mal selbst
trainieren, und beide sind häufige Fehlerquellen in echten Projekten.

*Was passiert eigentlich, wenn wir zweimal hintereinander ableiten?*

% Live-Frage: Vermutung einholen, bevor die Zelle ausgeführt wird
% (typische falsche Antwort: es bleibt bei 6).

````{admonition} Mini-Übung
:class: tip
Was gibt die folgende Schleife aus? Stellen Sie zuerst eine Vermutung auf und
prüfen Sie dann in der Code-Zelle.

```python
x = torch.tensor(3.0, requires_grad=True)
for i in range(2):
    y = x**2
    y.backward()
    print(x.grad)
```
````

```{code-cell} python
# Code-Zelle
```

```{admonition} Lösung
:class: tip
:class: dropdown
Die Ausgabe ist `tensor(6.)` und dann `tensor(12.)`. PyTorch überschreibt
`.grad` nicht, sondern **addiert** jeden neu berechneten Gradienten auf den
alten Wert. Nach dem zweiten Durchlauf steht deshalb 6 + 6 = 12 im Attribut.
```

PyTorch **akkumuliert** Gradienten: Jeder Aufruf von `backward()` addiert auf
das, was bereits in `.grad` steht. Schauen wir uns das direkt an:

```{code-cell} python
x = torch.tensor(3.0, requires_grad=True)
for i in range(2):
    y = x**2
    y.backward()
    print(x.grad)
```

Beim zweiten Durchlauf steht 12 statt 6 im Gradienten, obwohl sich an der
Funktion nichts geändert hat. In einer Trainingsschleife wäre das fatal, denn
die Gradienten aller bisherigen Schritte würden sich aufsummieren und das
Training in eine falsche Richtung lenken. Vor jedem neuen `backward()` müssen
wir die Gradienten deshalb ausdrücklich auf null setzen. Für einen einzelnen
Tensor erledigt das die Methode `grad.zero_()`; der Unterstrich am Ende ist die
PyTorch-Konvention für Methoden, die den Tensor direkt an Ort und Stelle
verändern:

% Lücke (Präsenz): die Zeile `x.grad.zero_()`

```{code-cell} python
x.grad.zero_()
print(x.grad)
```

Die zweite Eigenheit betrifft den Update-Schritt selbst. Aus Kapitel 3 kennen
wir das Rezept des Gradientenabstiegs: neuer Wert gleich alter Wert minus
Learning Rate mal Gradient. Diese Subtraktion ist aber auch nur eine Rechnung
mit einem beobachteten Tensor, und Autograd würde sie pflichtbewusst in den
Berechnungsgraphen aufnehmen, obwohl sie mit der Loss nichts zu tun hat. Für
solche Buchhaltungsschritte gibt es den Kontext `torch.no_grad()`: Alles, was
darin passiert, wird nicht protokolliert. Ein einzelner Abstiegsschritt für
unser Beispiel $y = x^2$ mit der Learning Rate 0,1 sieht so aus:

% Lücke (Präsenz): der Update-Schritt innerhalb von torch.no_grad()

```{code-cell} python
x = torch.tensor(3.0, requires_grad=True)
y = x**2
y.backward()

with torch.no_grad():
    x -= 0.1 * x.grad
x.grad.zero_()

print(x)
```

Aus 3 wird 2,4, ein Schritt bergab in Richtung des Minimums bei null, und dank
`torch.no_grad()` bleibt der Berechnungsgraph davon unberührt. Damit ist die
Arbeitsteilung klar. Autograd übernimmt genau eine Aufgabe, diese aber
perfekt: die Gradienten. Unsere Aufgabe bleibt alles andere, nämlich die
Architektur des Netzes, die Wahl der Loss, die Wahl der Learning Rate und die
Trainingsschleife aus Vorwärtsrechnung, `backward()`, Update-Schritt und
Nullsetzen der Gradienten. Genau diese Schleife bauen Sie jetzt in der
Studio-Aufgabe zum ersten Mal komplett selbst.

## Aufgaben

Zum Abschluss wenden wir alle Bausteine dieser Sektion in einer
zusammenhängenden Aufgabe an. Sie ist für die Bearbeitung in Kleingruppen
im Studio-Teil des Termins gedacht (etwa 15 Minuten); die Teilaufgaben folgen
genau der Reihenfolge, in der wir die Bausteine kennengelernt haben.

````{admonition} Übung 4.1 (✩✩): Federkennlinie mit Autograd
:class: tip
Im Prüflabor wurde unsere Schraubenfeder vermessen: zu 20 Auslenkungen $s$
(in Millimetern) liegt jeweils die gemessene Rückstellkraft $F$ (in Newton)
vor. Das physikalische Modell ist eine Gerade $F = k \cdot s + F_0$ mit der
Federsteifigkeit $k$ und der Vorspannkraft $F_0$. Beide Parameter sollen per
Gradientenabstieg aus den Messdaten bestimmt werden. Verwenden Sie das
folgende Gerüst:

```python
import torch

torch.manual_seed(42)
s = torch.linspace(0, 10, 20)                  # Auslenkung in mm
F = 0.8 * s + 2.0 + 0.4 * torch.randn(20)     # gemessene Kraft in N

# Teilaufgabe 1: Parameter anlegen
k = ...   # IHR CODE HIER
F0 = ...  # IHR CODE HIER

# Teilaufgabe 2: Vorhersage und Loss
# IHR CODE HIER

# Teilaufgabe 3: Gradienten berechnen und ausgeben
# IHR CODE HIER

# Teilaufgabe 4: ein Update-Schritt (Learning Rate 0.01), Gradienten nullen
# IHR CODE HIER
```

1. Legen Sie die Parameter `k` und `F0` als Tensoren mit dem Startwert 0 an,
   sodass Autograd sie beobachtet.
2. Berechnen Sie die Vorhersage $F_{\text{pred}} = k \cdot s + F_0$ und als
   Loss den mittleren quadratischen Fehler
   `loss = ((F_pred - F)**2).mean()`. Lassen Sie sich den Loss-Wert ausgeben.
3. Berechnen Sie die Gradienten und lassen Sie sich `k.grad` und `F0.grad`
   ausgeben. Deuten Sie die Vorzeichen: In welche Richtung wird der nächste
   Update-Schritt die beiden Parameter verändern, und ist das physikalisch
   plausibel?
4. Führen Sie einen Update-Schritt mit der Learning Rate 0,01 aus und setzen
   Sie anschließend die Gradienten auf null. Woran müssen Sie beim
   Update-Schritt denken?
5. **Zusatzaufgabe:** Packen Sie die Teilaufgaben 2 bis 4 in eine Schleife mit
   1000 Durchläufen. Lassen Sie sich am Ende `k` und `F0` ausgeben und
   vergleichen Sie mit den wahren Werten, die in der Datenerzeugung stecken.
   Stellen Sie die Messpunkte und die gefittete Gerade mit Plotly dar.
````

```{code-cell} python
# Code-Zelle
```

````{admonition} Lösung
:class: tip
:class: dropdown
```python
import torch

torch.manual_seed(42)
s = torch.linspace(0, 10, 20)                  # Auslenkung in mm
F = 0.8 * s + 2.0 + 0.4 * torch.randn(20)     # gemessene Kraft in N

# Teilaufgabe 1: Parameter anlegen
k = torch.tensor(0.0, requires_grad=True)
F0 = torch.tensor(0.0, requires_grad=True)

# Teilaufgabe 2: Vorhersage und Loss
F_pred = k * s + F0
loss = ((F_pred - F)**2).mean()
print(f'Loss: {loss.item():.4f}')

# Teilaufgabe 3: Gradienten berechnen und ausgeben
loss.backward()
print(f'Gradient k:  {k.grad.item():.4f}')
print(f'Gradient F0: {F0.grad.item():.4f}')

# Teilaufgabe 4: ein Update-Schritt, Gradienten nullen
with torch.no_grad():
    k -= 0.01 * k.grad
    F0 -= 0.01 * F0.grad
k.grad.zero_()
F0.grad.zero_()
```

Zu Teilaufgabe 3: Beide Gradienten sind negativ. Der Update-Schritt
subtrahiert die Gradienten, negative Gradienten vergrößern also die
Parameter. Das ist plausibel: Mit den Startwerten null sagt das Modell
überall die Kraft null voraus und liegt damit unter allen Messwerten, also
müssen sowohl die Steigung `k` als auch die Vorspannkraft `F0` wachsen.

Zu Teilaufgabe 4: Der Update-Schritt gehört in einen
`torch.no_grad()`-Block, damit er nicht selbst im Berechnungsgraphen landet.
Ohne das anschließende Nullsetzen würden sich die Gradienten über die
Schleifendurchläufe der Zusatzaufgabe aufsummieren und der Fit misslingt.

```python
# Teilaufgabe 5 (Zusatz): Trainingsschleife und Plot
import plotly.graph_objects as go

torch.manual_seed(42)
s = torch.linspace(0, 10, 20)
F = 0.8 * s + 2.0 + 0.4 * torch.randn(20)

k = torch.tensor(0.0, requires_grad=True)
F0 = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.01

for schritt in range(1000):
    F_pred = k * s + F0
    loss = ((F_pred - F)**2).mean()
    loss.backward()
    with torch.no_grad():
        k -= learning_rate * k.grad
        F0 -= learning_rate * F0.grad
    k.grad.zero_()
    F0.grad.zero_()

print(f'k  = {k.item():.4f} N/mm')
print(f'F0 = {F0.item():.4f} N')

fig = go.Figure()
fig.add_scatter(x=s.numpy(), y=F.numpy(), mode='markers', name='Messwerte')
fig.add_scatter(x=s.numpy(), y=(k * s + F0).detach().numpy(),
                mode='lines', name='gefittete Gerade')
fig.update_layout(xaxis_title='Auslenkung s in mm',
                  yaxis_title='Kraft F in N',
                  title='Federkennlinie: Messwerte und Fit')
fig.show()
```

Die gefitteten Parameter liegen nahe an den wahren Werten aus der
Datenerzeugung (Federsteifigkeit 0,8 N/mm und Vorspannkraft 2 N); wegen des
Messrauschens treffen sie sie nicht exakt, und das sollen sie auch nicht. Vor
dem Plotten muss die Vorhersage mit `.detach()` aus dem Berechnungsgraphen
gelöst werden, denn `.numpy()` funktioniert nur für Tensoren, die Autograd
nicht beobachtet. Wer diese Schleife verstanden hat, hat das Grundgerüst
jedes PyTorch-Trainings verstanden. In der nächsten Sektion lassen wir uns
genau diese Buchhaltung von PyTorch abnehmen.
````

## Zusammenfassung und Ausblick

In dieser Sektion haben wir die beiden Grundbausteine von PyTorch
kennengelernt. Tensoren speichern unsere Daten als Zahlenschemata mit beliebig
vielen Dimensionen und lassen sich über `torch.from_numpy` und `.numpy()`
verlustfrei mit NumPy austauschen. Autograd protokolliert jede Rechnung mit
beobachteten Tensoren im Berechnungsgraphen und liefert auf `backward()` alle
Gradienten frei Haus; am Mini-Netz aus Kapitel 3 haben wir nachgeprüft, dass
sie Ziffer für Ziffer mit unserer Handrechnung übereinstimmen. Zwei Dinge
bleiben dabei unsere Verantwortung: die Gradienten vor jedem neuen
`backward()` auf null zu setzen und den Update-Schritt in `torch.no_grad()` zu
verpacken. In der Studio-Aufgabe haben Sie daraus bereits eine vollständige
Trainingsschleife gebaut. *Aber müssen wir diese Buchhaltung wirklich für
jedes Gewicht einzeln erledigen?* Natürlich nicht: In der nächsten Sektion
übernehmen `torch.nn` und die Optimierer aus `torch.optim` genau diese Arbeit,
und aus unserer handgeschriebenen Schleife wird das Standardrezept jedes
PyTorch-Trainings.
