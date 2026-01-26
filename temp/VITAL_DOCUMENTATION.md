# Implémentation Vital - Overview

## Contexte

Passage du synthétiseur Surge XT à Vital avec une nouvelle approche : utiliser des presets réels au lieu de paramètres aléatoires pour générer des sons plus représentatifs de l'usage réel.

## Problèmes Principaux

### 1. Incompatibilité des formats
Les presets Surge étaient en .fxp (VST2) alors que pedalboard ne supporte que VST3. Solution : passage à Vital dont les presets sont en JSON (.vital), donc facilement parsables.

### 2. Mapping des 750 paramètres Vital
Les noms de paramètres dans les presets JSON ne correspondent pas toujours à ceux de l'API VST3. Création de trois fichiers :
- **vital_details.py** : ranges min/max de chaque paramètre (extraits du code source)
- **vital_preset_converter.py** : convertit .vital → paramètres plugin (gère ~100 cas spéciaux de nommage)
- **vital_param_spec.py** : définit l'encodage pour l'entraînement (continus, catégoriels, discrets)

### 3. Raw values vs semantic values
Pedalboard "devine" les plages de valeurs (Hz, dB, ms) ce qui introduit des erreurs. Solution : utiliser directement `raw_value` (normalisé [0,1]) pour un contrôle précis.

### 4. Wavetables et samples custom
15% des presets utilisent des wavetables/samples custom non paramétrables. Script de filtrage automatique qui garde :
- Presets avec oscillateurs de base uniquement (15%)
- Presets avec bruit blanc (70%)
- Rejette les wavetables/samples custom (15%)

## Génération de Dataset

### Approche "sampling around presets"
1. Charger un preset existant
2. Ajouter du bruit gaussien (std=0.1) sur les **paramètres continus uniquement**
3. Garder fixes les **paramètres catégoriels** (type filtre, forme d'onde) pour préserver la structure sonore

Avantages : sons réalistes, diversité autour de chaque preset, évite les combinaisons non-musicales.

### Optimisations
**Problème** : Charger un preset prend 2-3 secondes.
**Solution** : Système de cache (pickle) pour précharger tous les presets. Gain : 40 min → 1 sec pour 1000 presets.

## Résultats

**Premier entraînement** (100k samples, 200k steps, 3 jours sur L40S) :

| Métrique | Valeur | Baseline |
|----------|--------|----------|
| MSS | 14.40 | 28.06 |
| wMFCC | 27.77 | 36.34 |
| SOT | 0.148 | 0.439 |

Amélioration significative mais convergence non complète → dataset 1M samples en cours de génération.

**Performance génération** : ~104 samples/cpu/h sur 12 cores → dataset 1M en ~80h (~300€ sur OVH)

## Limitations

- 15% des presets réels inutilisables (wavetables custom)
- Effets temporels (delay, reverb) difficiles à capturer
- LFOs rapides créent des variations intra-note

## Pistes futures

- Dataset 1M samples pour améliorer convergence
- Pré-entraînement SSL de l'encodeur audio
- Encoder les wavetables custom en vecteurs latents
- Modèle multi-synthés conditionné sur le type de VST
