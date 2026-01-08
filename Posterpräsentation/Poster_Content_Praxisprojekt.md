# Inhalt & Gestaltung: Posterpräsentation (Praxisprojekt)

Diese Datei enthält Inhalte und Gestaltungs-Tipps für das wissenschaftliche Plakat. Die Inhalte basieren auf der Datenerfassung (Notebook 1.1) und den Qualitätsanalysen (Notebook 2.3 & 2.4), die den Kern des Praxisprojekts bilden.

---

## 🎨 Gestaltung & Didaktik (Wissenschaftliches Plakat)

Ein wissenschaftliches Plakat soll "Aufmerksamkeit erregen" und "Wissen in kurzer Zeit vermitteln".

### 1. Grundregeln
*   **Leserichtung**: Von oben links nach unten rechts (Spaltenlayout bevorzugt).
*   **Whitespace**: Mut zur Lücke! Überfrachtete Plakate werden ignoriert. Randabstände einhalten.
*   **Visuelle Hierarchie**:
    *   **Titel**: 70-100pt (muss aus 5m lesbar sein).
    *   **Headings**: 40-50pt.
    *   **Fließtext**: 24-32pt (Serifenlose Schrift wie Arial, Helvetica oder Roboto für gute Lesbarkeit).

### 2. Farbkonzept
Da es sich um Hydrogeochemie/Geologie handelt, empfiehlt sich ein **natürliches, professionelles Farbschema**:
*   **Primärfarbe**: Ein tiefes Blau (Wasser, Tiefe) oder Petrol.
*   **Akzentfarbe**: Ein gedecktes Orange oder Ocker (Gestein, Wärme/Temperatur).
*   **Hintergrund**: Sehr helles Grau oder Weiß. Kein dunkler Hintergrund (Druckkosten + Lesbarkeit).

### 3. Didaktischer Aufbau (Storytelling)
Erzählen Sie die Geschichte der Daten:
1.  **Woher** kommen sie? (Akquise)
2.  **Was** wurde gemacht? (Homogenisierung & Cleaning)
3.  **Wie gut** sind sie? (Qualitätskontrolle/IBE)
4.  **Was sehen** wir? (Erste Ergebnisse/Temperatur)

---

## 📝 Text-Inhalte (Vorschläge für die Sektionen)

Kopieren Sie diese Texte in die entsprechenden Textfelder der PPTX-Vorlage und passen Sie sie bei Bedarf an.

## 📝 Text-Inhalte (Angepasst auf 4 Sektionen)

Kopieren Sie diese Texte in die entsprechenden 4 Hauptbereiche Ihrer PPTX-Vorlage.

### 1. Einleitung & Motivation
> *Ziel: Kontext setzen. Warum machen wir das?*

**Inhalt:**
Die Analyse globaler hydrogeochemischer Daten ist essentiell für das Verständnis von Grundwasserleitern und geothermischen Potenzialen.
Ziel dieses Praxisprojekts war der Aufbau einer **homogenisierten "Big Data" Grundlage** aus heterogenen internationalen Quellen.
Herausforderungen dabei sind unterschiedliche Datenformate, Einheiten und Qualitätsstandards der Rohdaten. Das Projekt schafft die Basis für weiterführende Machine Learning-Analysen.

### 2. Methodik: Datenpipeline & Validierung
> *Kombiniert Data Acquisition (Nb 1.1) und Validierungs-Logik (Nb 2.4)*

**Prozess:**
*   **Datenquellen:** Aggregation von **13 internationalen Datenbanken** (u.a. USGS, ALPS, HESSEN, YUKON).
*   **Homogenisierung:** Entwicklung eines Python-Wrappers zur Vereinheitlichung auf **30 Ziel-Attribute** und Standard-Einheiten (mmol/L, °C).
*   **Validierungs-Ansatz:** Berechnung des **Ionenbilanzfehlers (IBE)** zur Qualitätskontrolle.
    *   Formel: $IBE = \frac{\sum Kationen - \sum Anionen}{\sum Kationen + \sum Anionen} \cdot 100$
    *   **Fokus-Ionen:** Na, Ca, Mg (Kationen) und Cl, HCO3, SO4 (Anionen).

### 3. Ergebnisse: Datenqualität & Exploration
> *Kombiniert IBE-Statistik (Nb 2.4) und Temperatur-Analyse (Nb 2.3)*

**Qualitäts-Status:**
*   Anwendung eines strikten Filters: Nur Datensätze mit **IBE < ± 5%** gelten als "Good Records".
*   Dies stellt sicher, dass nur chemisch valide Wasserproben für weitere Analysen genutzt werden.

**Explorative Datenanalyse (EDA):**
*   **Temperatur:** Auswertung der Temperaturverteilung über alle Datenbanken zur Identifikation geothermischer Anomalien.
*   **Verfügbarkeit:** Quantifizierung von "Missing Values" – nicht jede chemische Probe enthält Temperaturdaten, was die nutzbare Datenmenge für spezifische thermische Fragestellungen reduziert.

### 4. Fazit & Ausblick
> *Zusammenfassung des Praxisprojekts und Brücke zur Bachelorarbeit*

**Zusammenfassung:**
Es wurde erfolgreich eine skalierbare ETL-Pipeline (Extract, Transform, Load) implementiert. Das Ergebnis ist eine **validierte, chemisch neutrale Datenbasis** aus Millionen von Rohdatenpunkten.

**Ausblick (Bachelorarbeit):**
Die bereinigten "Good Records" bilden das Fundament für das **Machine Learning (Self-Organizing Maps)**. Im nächsten Schritt werden diese Daten genutzt, um feine hydrogeochemische Cluster und Gesteins-Wasser-Interaktionen zu modellieren, ohne durch "Daten-Rauschen" fehlerhafter Messungen verfälscht zu werden.

---

## 📊 Abbildungs-Ideen (Platzhalter)

Fügen Sie an den entsprechenden Stellen Screenshots aus den Notebooks ein:

1.  **Bei Methodik:** Ein Screenshot des `target_structure` Dataframes aus Notebook 1.1 (Liste der Attribute) oder der IBE-Formel.
2.  **Bei Ergebnisse:** Das Histogramm aus `Adjusted_Data_IBE_Report.pdf` (Gauß-Verteilung des Fehlers) und/oder ein Temperatur-Histogramm aus `Temperature_Analysis_Report.pdf` nebeneinander, um "Qualität" und "Daten" zu zeigen.
