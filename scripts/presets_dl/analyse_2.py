import json
import os
import rootutils
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
# Seuil pour considérer un paramètre comme "candidat à la suppression"
# 0.99 signifie : si 99% des presets ont la même valeur, on le marque comme ignorable.
CONSTANCY_THRESHOLD = 0.98 

root = rootutils.find_root(search_from=os.path.dirname(os.path.abspath(__file__)), indicator=".project-root")
plugin_dir = root.joinpath('data', 'presets', 'vital').as_posix()
scripts_dir = root.joinpath('scripts', 'presets_dl').as_posix()

# 1. Collecte des fichiers
all_vital_files = []
for root_dir, dirs, files in os.walk(plugin_dir):
    for file in files:
        if file.endswith('.vital'):
            all_vital_files.append(os.path.join(root_dir, file))

print(f"Found {len(all_vital_files)} .vital files.")

# 2. Extraction des données
# On utilise un dictionnaire de listes pour charger en mémoire
params_data = defaultdict(list)

# Pour s'assurer qu'on a le même nombre de lignes partout, on garde une trace des fichiers processés
processed_count = 0

for vital_file in tqdm(all_vital_files, desc="Reading presets"):
    try:
        with open(vital_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'settings' in data:
                settings = data['settings']
                # On itère sur tous les params trouvés dans ce fichier
                for key, value in settings.items():
                    # On ne garde que les nombres (les chaines comme "author" ne nous intéressent pas pour le son)
                    if isinstance(value, (int, float)):
                        params_data[key].append(value)
                processed_count += 1
    except json.JSONDecodeError:
        continue

print(f"Successfully processed {processed_count} presets.")

# 3. Analyse Statistique Avancée
analysis_results = []

print("Analyzing parameter relevance...")

for param_name, values in tqdm(params_data.items(), desc="Computing stats"):
    total = len(values)
    
    # Si un paramètre n'était pas présent dans tous les fichiers, on doit le noter
    # (Vital ajoute parfois des params dans les nouvelles versions)
    missing_count = processed_count - total
    
    # Si le paramètre est manquant dans trop de fichiers, on assume une valeur par défaut (souvent 0)
    # Pour l'analyse, on travaille sur les valeurs existantes
    if total == 0:
        continue

    # Compter les occurrences de chaque valeur
    counts = Counter(values)
    most_common_val, most_common_count = counts.most_common(1)[0]
    
    # Calcul de la "Dominance" (0.0 à 1.0)
    # 1.0 = Le paramètre a EXACTEMENT la même valeur dans tous les fichiers
    dominance_ratio = most_common_count / total
    
    unique_values_count = len(counts)
    
    # Décision de recommandation
    action = "KEEP"
    reason = "Variable"
    
    if unique_values_count == 1:
        action = "DELETE (Constant)"
        reason = f"Always {most_common_val}"
    elif dominance_ratio >= CONSTANCY_THRESHOLD:
        action = "DELETE (Near Constant)"
        reason = f"{most_common_val} in {dominance_ratio:.1%} of cases"
    elif unique_values_count < 5 and dominance_ratio > 0.90:
        action = "REVIEW"
        reason = "Low variance"

    analysis_results.append({
        'parameter': param_name,
        'action_recommendation': action,
        'dominance_ratio': round(dominance_ratio, 4),
        'most_common_value': most_common_val,
        'unique_values_count': unique_values_count,
        'min': min(values),
        'max': max(values),
        'reason': reason
    })

# 4. Création du rapport avec Pandas
df = pd.DataFrame(analysis_results)

# Trier par "Dominance" (les plus inutiles en premier)
df = df.sort_values(by='dominance_ratio', ascending=False)

# Sauvegarder en CSV (plus lisible pour Excel/Google Sheets)
csv_path = os.path.join(scripts_dir, 'vital_optimization_report.csv')
df.to_csv(csv_path, index=False)

# Sauvegarder en JSON
json_path = os.path.join(scripts_dir, 'vital_optimization_report.json')
df.to_json(json_path, orient='records', indent=4)

# 5. Résumé Console
print("-" * 30)
print(f"ANALYSIS COMPLETE.")
print(f"Report saved to: {csv_path}")
print("-" * 30)
print("SUMMARY:")
print(f"Total parameters analyzed: {len(df)}")
print(f"Recommended to DELETE (Constant 100%): {len(df[df['unique_values_count'] == 1])}")
print(f"Recommended to DELETE (Near Constant > {CONSTANCY_THRESHOLD*100}%): {len(df[(df['dominance_ratio'] >= CONSTANCY_THRESHOLD) & (df['unique_values_count'] > 1)])}")
print(f"Parameters to KEEP: {len(df[df['dominance_ratio'] < CONSTANCY_THRESHOLD])}")

# Afficher le top 10 des paramètres les plus inutiles qui ne sont pas constants à 100%
print("-" * 30)
print(f"Top 5 'Near Constant' parameters (Safe to remove?):")
near_constant = df[(df['dominance_ratio'] < 1.0) & (df['dominance_ratio'] > 0.95)].head(5)
print(near_constant[['parameter', 'dominance_ratio', 'reason']].to_string(index=False))