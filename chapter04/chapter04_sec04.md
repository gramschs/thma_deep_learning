---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Übungen

Mit Tensoren, Autograd, der Trainingsschleife und der Datenversorgung über Dataset und DataLoader ist der PyTorch-Werkzeugkasten dieser Woche komplett. Die folgenden Übungen führen vom Fingertraining mit Tensoren bis zur Fehlersuche in einer fremden Trainingsschleife, also genau der Situation, in der Sie in der Projektphase am häufigsten stecken werden. Übung 4.3 schlägt dabei bewusst die Brücke zurück zu Kapitel 3: Dasselbe Netz, einmal von Hand in NumPy und einmal in PyTorch, sollte dieselben Ergebnisse liefern. Zur Erinnerung: Übungen mit zwei Sternen entsprechen dem Klausurniveau.

```{admonition} Übung 4.1 (✩)
:class: tip
Gegeben ist das NumPy-Array `messung = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])`.

1. Konvertieren Sie `messung` in einen PyTorch-Tensor mit dem Datentyp `float32` und geben Sie Shape und dtype aus.
2. Berechnen Sie den Mittelwert jeder Spalte (das Ergebnis hat den Shape `(3,)`).
3. Konvertieren Sie das Ergebnis zurück in ein NumPy-Array.
4. Erzeugen Sie mit `reshape` aus dem Tensor aus Schritt 1 einen Tensor der Form `(3, 2)` und geben Sie ihn aus. Was passiert stattdessen bei `reshape(4, 2)` und warum?
```

```{code-cell} python
# IHR CODE HIER

```

````{admonition} Lösung
:class: tip
:class: dropdown
```python
import numpy as np
import torch

messung = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 1. Konvertierung mit float32
tensor = torch.from_numpy(messung).float()
print(tensor.shape, tensor.dtype)   # torch.Size([2, 3]) torch.float32

# 2. Spaltenmittelwerte: Mittelung über die Zeilen-Dimension (dim=0)
mittel = tensor.mean(dim=0)
print(mittel)                       # tensor([2.5000, 3.5000, 4.5000])

# 3. Zurück nach NumPy
mittel_numpy = mittel.numpy()
print(type(mittel_numpy))

# 4. Umformen
print(tensor.reshape(3, 2))
```

Zu Schritt 4: `reshape(3, 2)` ordnet die sechs Werte zeilenweise in drei Zeilen mit je zwei Werten an. `reshape(4, 2)` bricht dagegen mit einer Fehlermeldung ab, weil dafür acht Werte nötig wären, der Tensor aber nur sechs Elemente enthält. Die Gesamtzahl der Elemente muss bei `reshape` erhalten bleiben.
````

```{admonition} Übung 4.2 (✩)
:class: tip
In Übung 3.4 haben Sie den Gradienten der Funktion

$$L(w_1, w_2) = \bigl(w_2 \cdot \tanh(w_1 \cdot x) - t\bigr)^2$$

an der Stelle $x = 1$, $t = 1$, $w_1 = 0{,}5$, $w_2 = 2$ von Hand berechnet.

1. Bilden Sie dieselbe Rechnung mit PyTorch-Tensoren nach: Legen Sie `w1` und `w2` mit `requires_grad=True` an, bauen Sie `L` auf und rufen Sie `backward()` auf.
2. Geben Sie `w1.grad` und `w2.grad` aus und vergleichen Sie mit Ihrer Handrechnung aus Übung 3.4.
3. Rufen Sie `backward()` ein zweites Mal auf derselben neu aufgebauten Loss auf, ohne die Gradienten zu nullen. Erklären Sie die ausgegebenen Werte.
```

```{code-cell} python
# IHR CODE HIER

```

````{admonition} Lösung
:class: tip
:class: dropdown
```python
import torch

x = torch.tensor(1.0)
t = torch.tensor(1.0)
w1 = torch.tensor(0.5, requires_grad=True)
w2 = torch.tensor(2.0, requires_grad=True)

# 1. Forward Pass und Backward Pass
L = (w2 * torch.tanh(w1 * x) - t) ** 2
L.backward()

# 2. Gradienten ausgeben
print(w1.grad)   # ≈ -0.2383
print(w2.grad)   # ≈ -0.0700

# 3. Loss neu aufbauen und erneut backward()
L = (w2 * torch.tanh(w1 * x) - t) ** 2
L.backward()
print(w1.grad)   # ≈ -0.4767, also der doppelte Wert
print(w2.grad)   # ≈ -0.1401, also der doppelte Wert
```

Zur Kontrolle die Handrechnung: Mit $h = \tanh(0{,}5) \approx 0{,}4621$ und $y = w_2 h \approx 0{,}9242$ ist der gemeinsame Faktor $2(y - t) \approx -0{,}1515$. Daraus folgt $\partial L / \partial w_2 = 2(y - t) \cdot h \approx -0{,}0700$ und $\partial L / \partial w_1 = 2(y - t) \cdot w_2 \cdot (1 - h^2) \cdot x \approx -0{,}2383$ (kleine Abweichungen je nach Rundung in der Handrechnung sind normal).

Zu Schritt 3: `backward()` addiert die Gradienten auf die vorhandenen Werte in `.grad`, statt sie zu überschreiben. Nach dem zweiten Aufruf steht deshalb jeweils der doppelte Gradient im Attribut. Genau deshalb gehört `optimizer.zero_grad()` in jede Trainingsschleife.
````

```{admonition} Übung 4.3 (✩✩)
:class: tip
In Übung 3.5 haben Sie ein Netz mit einer verborgenen Schicht (1 Eingang, 3 verborgene Neuronen mit ReLU, 1 Ausgang) in reinem NumPy implementiert und auf den dort erzeugten Daten trainiert.

1. Bauen Sie dasselbe Netz als `nn.Module` in PyTorch nach.
2. Erzeugen Sie dieselben Trainingsdaten wie in Übung 3.5 (gleiche Formel, Seed 42) als Tensoren der Form `(n, 1)`.
3. Trainieren Sie das Netz 500 Epochen mit `nn.MSELoss` und `torch.optim.SGD` (Learning Rate wie in Übung 3.5) auf dem vollen Datensatz.
4. Vergleichen Sie die finale Loss mit Ihrem NumPy-Ergebnis aus Übung 3.5. Erwarten Sie exakt dieselben Zahlen? Begründen Sie.
```

```{code-cell} python
# IHR CODE HIER

```

````{admonition} Lösung
:class: tip
:class: dropdown
```python
import torch
import torch.nn as nn

# 2. Daten wie in Übung 3.5 (Beispiel: verrauschte Parabel)
torch.manual_seed(42)
n = 100
x = torch.rand(n, 1) * 4.0 - 2.0
y = x**2 + torch.randn(n, 1) * 0.1

# 1. Netz 1-3-1 mit ReLU
class KleinesNetz(nn.Module):
    def __init__(self):
        super().__init__()
        self.schichten = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )

    def forward(self, x):
        return self.schichten(x)

torch.manual_seed(42)
modell = KleinesNetz()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(modell.parameters(), lr=0.05)

# 3. Training
for epoche in range(500):
    y_pred = modell(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Finale Loss: {loss.item():.4f}")
```

Zu Schritt 4: Die finale Loss sollte in derselben Größenordnung liegen wie in Übung 3.5, exakt gleiche Zahlen sind aber nicht zu erwarten. Der Grund ist die Initialisierung: PyTorch initialisiert die Gewichte eines `nn.Linear` nach einem eigenen Schema, das sich von unserer NumPy-Initialisierung unterscheidet, selbst wenn beide denselben Seed verwenden. Gleiche Seeds garantieren Reproduzierbarkeit innerhalb eines Werkzeugs, nicht zwischen verschiedenen Werkzeugen. Vergleichbar ist deshalb das Fehlerniveau nach dem Training, nicht der einzelne Gewichtswert.
````

```{admonition} Übung 4.4 (✩✩)
:class: tip
Beim Drehen verschleißt die Schneidkante des Werkzeugs. Wir wollen die Verschleißmarkenbreite aus den Prozessparametern vorhersagen. Erzeugen Sie zunächst den synthetischen Datensatz:

1. Erzeugen Sie mit Seed 42 einen Datensatz aus 300 Versuchen: Schnittgeschwindigkeit `v_c` gleichverteilt in [50, 200] m/min, Vorschub `f` gleichverteilt in [0,1, 0,5] mm/U, Eingriffszeit `t` gleichverteilt in [1, 30] min. Die Zielgröße ist `vb = 0.002 * t * (v_c / 100)**1.5 * (f / 0.2)**0.5` plus normalverteiltes Rauschen mit Standardabweichung 0,01.
2. Stapeln Sie die drei Eingangsgrößen zu einer Matrix `X` der Form `(300, 3)` und standardisieren Sie jede Spalte (Mittelwert abziehen, durch Standardabweichung teilen).
3. Verpacken Sie die Daten in ein eigenes `Dataset` und einen `DataLoader` mit Batch Size 32 und Shuffling.
4. Trainieren Sie ein Netz mit einer verborgenen Schicht (3 Eingänge, 32 Neuronen, ReLU, 1 Ausgang) über 200 Epochen mit Adam (Learning Rate 0,01).
5. Protokollieren Sie die mittlere Loss pro Epoche und stellen Sie den Verlauf mit Plotly dar (logarithmische y-Achse).
```

```{code-cell} python
# IHR CODE HIER

```

````{admonition} Lösung
:class: tip
:class: dropdown
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go

# 1. Synthetischer Verschleiß-Datensatz
torch.manual_seed(42)
n = 300
v_c = torch.rand(n, 1) * 150.0 + 50.0     # m/min
f = torch.rand(n, 1) * 0.4 + 0.1          # mm/U
t = torch.rand(n, 1) * 29.0 + 1.0         # min
vb = 0.002 * t * (v_c / 100)**1.5 * (f / 0.2)**0.5
vb = vb + torch.randn(n, 1) * 0.01        # Messrauschen

# 2. Merkmalsmatrix bauen und standardisieren
X = torch.cat([v_c, f, t], dim=1)
X = (X - X.mean(dim=0)) / X.std(dim=0)

# 3. Dataset und DataLoader
class VerschleissDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = VerschleissDataset(X, vb)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Modell, Loss, Optimizer
torch.manual_seed(42)
modell = nn.Sequential(
    nn.Linear(3, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modell.parameters(), lr=0.01)

# 5. Training mit Protokollierung
loss_verlauf = []
for epoche in range(200):
    loss_summe = 0.0
    for X_batch, y_batch in loader:
        y_pred = modell(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_summe += loss.item() * len(X_batch)
    loss_verlauf.append(loss_summe / len(dataset))

fig = go.Figure()
fig.add_trace(go.Scatter(y=loss_verlauf, mode="lines", name="Trainings-Loss"))
fig.update_layout(xaxis_title="Epoche",
                  yaxis_title="Loss (MSE)",
                  yaxis_type="log")
fig.show()
```

Anmerkungen: Die Standardisierung in Schritt 2 ist hier wichtig, weil die drei Eingangsgrößen völlig verschiedene Größenordnungen haben (Schnittgeschwindigkeit bis 200, Vorschub unter 1). Ohne Standardisierung dominieren die großen Werte die ersten Schichten und das Training kommt nur zäh voran. Hier zeigt sich auch, warum `nn.Sequential` ohne eigene Klasse ausreicht, solange das Netz eine reine Kette von Schichten ist.
````

````{admonition} Übung 4.5 (✩✩✩)
:class: tip
Ein Kommilitone bittet Sie um Hilfe: Sein Training "läuft ohne Fehlermeldung, aber das Netz lernt nichts". In der folgenden Trainingsschleife stecken **drei Fehler**. Die Daten `s` und `F` sind die Federdaten aus Sektion 4.2 mit den Shapes `(200, 1)`.

```python
torch.manual_seed(42)
modell = FederNetz()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modell.parameters(), lr=0.05)

for epoche in range(300):
    F_pred = modell(s)
    loss = criterion(F_pred, F.squeeze())
    optimizer.step()
    loss.backward()
```

1. Finden Sie alle drei Fehler.
2. Begründen Sie für jeden Fehler, welches Fehlverhalten er verursacht und warum keine Fehlermeldung erscheint (oder höchstens eine Warnung).
3. Schreiben Sie die korrigierte Schleife auf und überzeugen Sie sich durch Ausführen, dass die Loss jetzt fällt.
````

```{code-cell} python
# IHR CODE HIER

```

````{admonition} Lösung
:class: tip
:class: dropdown
Die drei Fehler:

**Fehler 1: Shape-Fehler durch `F.squeeze()`.** Die Vorhersage `F_pred` hat den Shape `(200, 1)`, das Ziel nach `squeeze()` aber `(200,)`. Broadcasting macht aus der Differenz eine Matrix der Form `(200, 200)`, und die MSELoss mittelt über 40.000 sinnlose Paarungen. PyTorch gibt dafür nur eine UserWarning aus, keinen Fehler. Das Training optimiert eine falsche Größe und die Vorhersagen bleiben unbrauchbar.

**Fehler 2: `optimizer.step()` vor `loss.backward()`.** Der Optimizer-Schritt wird ausgeführt, bevor die Gradienten dieser Epoche berechnet sind. In der ersten Epoche existieren noch gar keine Gradienten (der Schritt bewirkt nichts), danach wird stets mit den veralteten Gradienten der Vorepoche aktualisiert. Auch das läuft ohne Fehlermeldung durch.

**Fehler 3: `optimizer.zero_grad()` fehlt ganz.** Die Gradienten aller Epochen summieren sich auf. Zusammen mit Fehler 2 aktualisiert die Schleife also mit der ständig wachsenden Summe veralteter Gradienten.

Die korrigierte Schleife:

```python
torch.manual_seed(42)
modell = FederNetz()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modell.parameters(), lr=0.05)

for epoche in range(300):
    F_pred = modell(s)
    loss = criterion(F_pred, F)      # Shapes (200, 1) und (200, 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Die allgemeine Lehre aus dieser Übung: Die gefährlichsten Fehler im Deep Learning sind nicht die, die das Programm abstürzen lassen, sondern die, die es stillschweigend das Falsche optimieren lassen. Ein Blick auf die Shapes und auf den Loss-Verlauf gehört deshalb zu jeder Fehlersuche.
````

## Zusammenfassung und Ausblick

In Woche 4 ist aus der Handrechnung von Kapitel 3 ein vollwertiges PyTorch-Training geworden: Tensoren als Datenstruktur, Autograd für die Gradienten, die Trainingsschleife als Herzstück und Dataset mit DataLoader für die Datenversorgung in Batches. Die Übungen haben gezeigt, dass dieselben Konzepte tragen, egal ob wir sie von Hand, in NumPy oder in PyTorch umsetzen, und dass die häufigsten Fehler stille Fehler sind. Damit läuft unser Training. In Kapitel 5 sorgen wir dafür, dass es gut läuft: Wir lernen, die Learning Rate zu wählen, das Training mit Validierungsdaten zu überwachen, Overfitting zu erkennen und kranke Lernkurven zu diagnostizieren.
