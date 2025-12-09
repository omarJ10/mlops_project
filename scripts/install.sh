#!/bin/bash

# Script d'installation pour Atelier 3
# Ce script installe Make et les dépendances nécessaires

echo "================================================"
echo "    Installation - Atelier 3 MLOps"
echo "================================================"

# Vérifier si make est installé
if ! command -v make &> /dev/null; then
    echo "[INFO] Installation de make..."
    sudo apt update
    sudo apt install -y make
    echo "[OK] Make installé!"
else
    echo "[OK] Make déjà installé"
fi

# Vérifier Python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 non trouvé! Veuillez l'installer."
    exit 1
else
    echo "[OK] Python3 trouvé: $(python3 --version)"
fi

# Vérifier pip3
if ! command -v pip3 &> /dev/null; then
    echo "[INFO] Installation de pip3..."
    sudo apt install -y python3-pip
    echo "[OK] pip3 installé!"
else
    echo "[OK] pip3 déjà installé"
fi

echo ""
echo "================================================"
echo "[OK] Installation terminée!"
echo "================================================"
echo ""
echo "Prochaines étapes:"
echo "  1. make install        # Installer les dépendances Python"
echo "  2. make test          # Tester le projet"
echo "  3. make pipeline      # Exécuter le pipeline ML"
echo "  4. make help          # Voir toutes les commandes"
echo ""
