# Spécifications : Distillation de caractéristiques via CLAP

## 1. Contexte et Problématique

Nous travaillons sur l'**inversion de synthèse** pour le synthétiseur **Vital**, en nous basant sur l'architecture de **Hayes et al. (2025)**.

- **L'existant :** Un encodeur **AST** (Audio Spectrogram Transformer) extrait des caractéristiques pour conditionner un modèle de **Flow Matching** qui prédit les paramètres du synthé.
- **Le problème :** Le modèle actuel est "trop habitué" à la propreté des sons numériques de Vital. Quand on lui soumet un instrument réel (OOD - Out-of-Distribution), il échoue.
- **La solution :** Utiliser la **Feature Distillation**. On va forcer notre encodeur (l'Élève) à imiter les représentations d'un modèle expert (le Professeur) qui comprend l'audio réel.

---

## 2. Le Modèle Choisi : `laion/clap-htsat-fused`

- **Pourquoi :** C'est un modèle SOTA (State of the Art) entraîné sur des millions de sons réels à **48 kHz**. Il possède une compréhension sémantique profonde du timbre que notre modèle n'a pas.
- **Rôle :** Il sert de **Professeur gelé**. On ne l'entraîne pas, on utilise seulement ses sorties (embeddings).

---

## 3. Tâches d'implémentation (Le "To-Do")

### A. Pipeline de données (Dual-branch)

Le développeur doit créer deux branches de prétraitement pour chaque échantillon audio :

1. **Branche Élève (AST) :** Garder le format actuel (souvent 44.1 kHz, 128 bandes Mel).
2. **Branche Professeur (CLAP) :** Convertir l'audio au format natif de CLAP (48 kHz, 64 bandes Mel). _Note : Les deux modèles doivent écouter le même segment audio de 4s._

### B. Architecture

1. **Chargement de CLAP :** Charger `laion/clap-htsat-fused` via Hugging Face. Passer le modèle en `.eval()` et désactiver les gradients (`requires_grad = False`).
2. **Projection Head :** Ajouter une couche linéaire (`nn.Linear`) à la sortie de notre AST pour aligner sa dimension sur celle de CLAP (512).

### C. Fonction de perte (Hybrid Loss)

Modifier la boucle d'entraînement pour calculer une perte combinée :

- : La perte actuelle de prédiction des paramètres.
- : L'écart entre l'embedding de notre AST et celui de CLAP.
- : Coefficient de pondération (commencer par 0.1).

---

## 4. Objectif final pour le code

À la fin de l'entraînement, l'encodeur AST doit être capable de générer des vecteurs de caractéristiques robustes, que le son soit synthétique ou réel. Cela permettra au modèle de Flow Matching de "comprendre" un instrument réel comme s'il s'agissait d'un preset Vital.
