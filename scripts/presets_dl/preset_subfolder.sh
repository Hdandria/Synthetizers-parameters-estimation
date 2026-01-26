#!/bin/bash

# Configuration
source="./data/presets/vital"
destination="./data/presets/vital_50"
number=50

# Verify source directory exists
if [ ! -d "$source" ]; then
  echo "Error: Source directory $source does not exist."
    exit 1
fi

# If destination exists, prompt for confirmation to overwrite
if [ -d "$destination" ]; then
  read -p "Destination directory $destination already exists. Overwrite? (y/n): " choice
  case "$choice" in
    y|Y ) echo "Overwriting $destination..."; rm -rf "$destination" ;;
    n|N ) echo "Operation cancelled."; exit 0 ;;
    * ) echo "Invalid choice. Operation cancelled."; exit 1 ;;
  esac
fi

echo "Moving $number random files from $source to $destination"

# Create destination directory if it doesn't exist
mkdir -p "$destination"

# Move random files from source to destination
find "$source" -maxdepth 1 -type f -print0 | shuf | head -n "$number" | xargs -0 -I {} mv {} "$destination"

echo "Moved $number files to $destination"