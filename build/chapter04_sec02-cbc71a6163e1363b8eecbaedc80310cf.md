---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Die Trainingsschleife

Eine progressive Fahrwerksfeder wird steifer, je weiter sie eingefedert ist: Die Kennlinie Kraft über Weg ist gekrümmt, eine Geradengleichung reicht nicht. Wir haben Messdaten aus dem Prüfstand und wollen ein neuronales Netz darauf trainieren, die Kennlinie vorherzusagen. Alles, was wir dafür brauchen, haben wir in den letzten Wochen einzeln kennengelernt: Modell, Loss, Gradient, Update. In dieser Sektion fügt sich das zu wenigen Zeilen Code zusammen, die in fast jedem Deep-Learning-Projekt der Welt nahezu identisch aussehen.

```{admonition} Lernziele
:class: attention
* [ ] Sie können ein neuronales Netz als nn.Module definieren.
* [ ] Sie können die Bestandteile der Trainingsschleife benennen und implementieren: Forward Pass, Loss, backward(), Optimizer-Schritt, Gradienten nullen.
* [ ] Sie können ein Training über mehrere Epochen ausführen und den Loss-Verlauf mit Plotly darstellen.
* [ ] Sie können typische Fehler in der Trainingsschleife erkennen (zum Beispiel fehlendes zero_grad).
```

## Die Messdaten

Wir erzeugen uns einen synthetischen Prüfstandsdatensatz: 200 Messpunkte einer progressiven Feder mit der wahren Kennlinie $F = k_1 s + k_3 s^3$ plus Messrauschen. In der Praxis kämen diese Daten aus einer CSV-Datei des Prüfstands; für das Skript erzeugen wir sie reproduzierbar mit einem Seed.

```{code-cell} python
import torch

torch.manual_seed(42)

n = 200
s = torch.rand(n, 1) * 100.0                  # Federweg in mm
F = 0.05 * s + 4e-5 * s**3                    # wahre Kennlinie in N
F = F + torch.randn(n, 1) * 2.0               # Messrauschen

print(s.shape, F.shape)
```

Beachten Sie die Shapes: Beide Tensoren haben die Form `(200, 1)`, also 200 Zeilen (Messpunkte) und eine Spalte (Merkmal beziehungsweise Zielgröße). Diese Konvention, Daten immer als 2D-Tensor mit einer Zeile pro Beispiel abzulegen, ziehen wir ab jetzt konsequent durch. Werfen wir einen Blick auf die Daten:

```{code-cell} python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=s.squeeze(), y=F.squeeze(),
                         mode="markers", name="Messdaten"))
fig.update_layout(xaxis_title="Federweg s in mm",
                  yaxis_title="Kraft F in N")
fig.show()
```

Die Krümmung ist deutlich sichtbar. *Wie bringen wir nun ein Netz dazu, diese Kennlinie zu lernen?*

## Das Modell: nn.Module

In Kapitel 3 haben wir Gewichtsmatrizen noch selbst angelegt und multipliziert. PyTorch bündelt diese Bausteine im Modul `torch.nn`. Eine vollvernetzte Schicht mit Gewichten und Bias heißt dort `nn.Linear`, und ein vollständiges Netz definieren wir als Klasse, die von `nn.Module` erbt:

```{code-cell} python
import torch.nn as nn

torch.manual_seed(42)

class FederNetz(nn.Module):
    def __init__(self):
        super().__init__()
        self.schichten = nn.Sequential(
            nn.Linear(1, 16),    # 1 Eingang (Federweg) auf 16 Neuronen
            nn.ReLU(),
            nn.Linear(16, 1),    # 16 Neuronen auf 1 Ausgang (Kraft)
        )

    def forward(self, x):
        return self.schichten(x)

modell = FederNetz()
print(modell)
```

Zwei Dinge erledigt `nn.Module` im Hintergrund für uns. Erstens legt jedes `nn.Linear` seine Gewichte und Bias-Werte selbst an, zufällig initialisiert und bereits mit `requires_grad=True` markiert. Wir bekommen alle Parameter gesammelt über `modell.parameters()`. Zweitens definiert die Methode `forward()` den **Forward Pass** (englisch: "Forward Pass"), also den Weg der Daten durch das Netz. Aufgerufen wird das Modell wie eine Funktion: `modell(s)` liefert die Vorhersagen.

```{code-cell} python
anzahl = sum(p.numel() for p in modell.parameters())
print(f"Anzahl trainierbarer Parameter: {anzahl}")
```

Unser Mini-Netz hat 49 Parameter: deutlich mehr, als wir von Hand ableiten möchten, und immer noch winzig im Vergleich zu echten Netzen.

## Loss und Optimizer

Für die Regression verwenden wir wie in Kapitel 3 den mittleren quadratischen Fehler, in PyTorch `nn.MSELoss`. Neu ist der **Optimizer** (englisch: "Optimizer"): das Objekt, das aus den Gradienten die eigentlichen Gewichtsänderungen macht. Genau diese Aufgabe hatte Autograd in der letzten Sektion ja ausdrücklich nicht übernommen. Die Idee dahinter kennen wir aus der Numerik: Auch das Newton-Verfahren tastet sich iterativ an eine Lösung heran, indem es lokale Ableitungsinformation in einen Schritt übersetzt. Der Gradientenabstieg ist die einfachere Variante, die nur erste Ableitungen verwendet, dafür aber mit Millionen Variablen umgehen kann.

```{code-cell} python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modell.parameters(), lr=0.05)
```

Der Optimizer bekommt beim Anlegen alle Parameter des Modells übergeben und die **Learning Rate** `lr`, also die Schrittweite. Wir verwenden hier den Optimizer Adam; was ihn vom reinen Gradientenabstieg unterscheidet und wie man die Learning Rate wählt, vertiefen wir in Kapitel 5. Für heute genügt: Adam mit einer Learning Rate zwischen 0,1 und 0,001 ist ein robuster Startpunkt.

## Die Schleife: fünf Zeilen, die alles tragen

Jetzt fügt sich alles zusammen. Ein Trainingsschritt besteht aus fünf Operationen, und eine **Epoche** (englisch: "Epoch") bezeichnet einen Durchlauf durch die gesamten Trainingsdaten:

1. **Forward Pass:** Vorhersagen des Modells berechnen.
2. **Loss:** Vorhersagen mit den Messwerten vergleichen.
3. **Gradienten nullen:** die akkumulierten Gradienten des letzten Schritts löschen.
4. **Backward Pass:** neue Gradienten mit `backward()` berechnen.
5. **Optimizer-Schritt:** Gewichte mit `optimizer.step()` anpassen.

```{admonition} Hinweis: Ihre Zahlen können abweichen
:class: warning
Wir setzen in allen Notebooks einen Seed, damit die Ergebnisse
reproduzierbar sind. Training auf einer GPU ist dennoch nie perfekt
deterministisch. Wenn Ihre Loss-Werte geringfügig von den hier
gezeigten abweichen, ist das normal und kein Fehler.
```

```{code-cell} python
torch.manual_seed(42)
modell = FederNetz()
optimizer = torch.optim.Adam(modell.parameters(), lr=0.05)

loss_verlauf = []

for epoche in range(300):
    F_pred = modell(s)                  # 1. Forward Pass
    loss = criterion(F_pred, F)         # 2. Loss berechnen
    optimizer.zero_grad()               # 3. Gradienten nullen
    loss.backward()                     # 4. Backward Pass
    optimizer.step()                    # 5. Gewichte anpassen

    loss_verlauf.append(loss.item())
    if (epoche + 1) % 50 == 0:
        print(f"Epoche {epoche + 1:3d}: Loss = {loss.item():.2f}")
```

Gehen wir die weniger offensichtlichen Zeilen durch. `optimizer.zero_grad()` ist die Konsequenz aus der letzten Sektion: Weil `backward()` Gradienten aufaddiert statt überschreibt, müssen wir vor jedem neuen Backward Pass aufräumen, sonst rechnen wir mit der Summe aller bisherigen Gradienten weiter. Und `loss.item()` holt den reinen Zahlenwert aus dem Tensor heraus; das trennt die Protokollierung sauber vom Berechnungsgraphen.

## Der Loss-Verlauf

Die Liste `loss_verlauf` ist unser wichtigstes Diagnoseinstrument. Aufgetragen über die Epochen zeigt sie, ob das Training überhaupt lernt:

```{code-cell} python
fig = go.Figure()
fig.add_trace(go.Scatter(y=loss_verlauf, mode="lines", name="Trainings-Loss"))
fig.update_layout(xaxis_title="Epoche",
                  yaxis_title="Loss (MSE)",
                  yaxis_type="log")
fig.show()
```

Wir verwenden eine logarithmische Achse, weil die Loss zu Beginn um Größenordnungen fällt und Details am Ende sonst nicht erkennbar wären. Der Verlauf, den wir hier sehen, ist der gesunde Normalfall: schneller Abfall am Anfang, dann ein allmähliches Abflachen. Wie man kranke Verläufe liest (Divergenz, Plateaus, Zickzack), ist ein Schwerpunkt von Kapitel 5.

Zum Abschluss prüfen wir, ob das Netz die Kennlinie tatsächlich getroffen hat. Für die Vorhersage brauchen wir keinen Berechnungsgraphen; der Kontext `torch.no_grad()` schaltet Autograd vorübergehend ab und spart damit Speicher und Rechenzeit:

```{code-cell} python
s_fein = torch.linspace(0, 100, 200).reshape(-1, 1)
with torch.no_grad():
    F_fein = modell(s_fein)

fig = go.Figure()
fig.add_trace(go.Scatter(x=s.squeeze(), y=F.squeeze(),
                         mode="markers", name="Messdaten"))
fig.add_trace(go.Scatter(x=s_fein.squeeze(), y=F_fein.squeeze(),
                         mode="lines", name="Vorhersage des Netzes"))
fig.update_layout(xaxis_title="Federweg s in mm",
                  yaxis_title="Kraft F in N")
fig.show()
```

Das Netz hat die progressive Kennlinie gelernt, ohne dass wir ihm die Form $k_1 s + k_3 s^3$ verraten haben. Genau das ist der Kern des überwachten Lernens: Die Struktur steckt in den Daten, nicht im Code.

## Typische Stolperfallen

Die Schleife ist kurz, aber jede Zeile trägt. Drei Fehler begegnen uns in der Praxis immer wieder.

**Fehlendes `zero_grad()`.** Ohne das Nullen akkumulieren sich die Gradienten über alle Epochen. Die effektiven Schritte werden immer größer, das Training wird unruhig oder divergiert. Tückisch daran: Der Code läuft ohne Fehlermeldung durch, nur die Loss verhält sich seltsam.

**Stille Shape-Fehler durch Broadcasting.** Hat die Vorhersage den Shape `(200, 1)`, die Zielgröße aber `(200,)`, dann macht Broadcasting aus der Differenz eine Matrix der Form `(200, 200)`, und die MSELoss mittelt fröhlich über 40.000 falsche Paarungen. Auch hier: keine Fehlermeldung, nur eine Warnung und ein Training, das nicht vorankommt. Deshalb die Konvention aus dem Datenabschnitt: Zielgrößen immer als `(n, 1)` ablegen und im Zweifel `print(x.shape)`.

**Vertauschte Reihenfolge.** `optimizer.step()` vor `loss.backward()` aufzurufen bedeutet, mit den Gradienten des letzten Schritts (oder mit Nullen) zu aktualisieren. Die Reihenfolge nullen, backward, step ist nicht verhandelbar.

```{admonition} Mini-Übung
:class: tip
In der folgenden Schleife fehlt genau eine Zeile. Was wird passieren, wenn wir sie ausführen: Fehlermeldung, Divergenz oder unauffälliges Verhalten? Ergänzen Sie anschließend die fehlende Zeile.
```

```{code-cell} python
torch.manual_seed(42)
modell_test = FederNetz()
optimizer_test = torch.optim.Adam(modell_test.parameters(), lr=0.05)

for epoche in range(100):
    F_pred = modell_test(s)
    loss = criterion(F_pred, F)
    loss.backward()
    optimizer_test.step()

print(f"Loss nach 100 Epochen: {loss.item():.2f}")
```

````{admonition} Lösung
:class: tip
:class: dropdown
Es fehlt `optimizer_test.zero_grad()`. Eine Fehlermeldung gibt es nicht: Die Schleife läuft durch, aber die Gradienten summieren sich über alle Epochen auf, die Schritte werden zu groß und die Loss bleibt deutlich schlechter als im sauberen Training (oder pendelt sichtbar hin und her). Korrekt lautet der Rumpf:

```python
for epoche in range(100):
    F_pred = modell_test(s)
    loss = criterion(F_pred, F)
    optimizer_test.zero_grad()
    loss.backward()
    optimizer_test.step()
```
````

## Zusammenfassung und Ausblick

In dieser Sektion ist aus den Einzelteilen der letzten Wochen ein vollständiges Training geworden. Wir definieren Netze als `nn.Module`, das seine Parameter selbst verwaltet, wählen Loss-Funktion und Optimizer, und die Schleife aus Forward Pass, Loss, `zero_grad()`, `backward()` und `step()` erledigt den Rest. Der Loss-Verlauf, mit Plotly aufgetragen, ist dabei unser Blick in den Maschinenraum. Einen Schönheitsfehler hat unser Training allerdings noch: Wir haben in jeder Epoche den kompletten Datensatz auf einmal durch das Netz geschickt. Bei 200 Messpunkten ist das kein Problem, bei 50.000 Bildern schon. Wie wir Daten häppchenweise in das Training füttern, klärt die nächste Sektion.
